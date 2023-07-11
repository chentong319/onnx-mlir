/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- DecomposeEinsum.cpp - Decompose Einsum op ----------------===//
//
// This file implements the decomposition of ONNXEinsumOp to simpler ops.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Transform/ONNX/MatrixDecompose.hpp"
// #include "src/Dialect/ONNX/ONNXOps/Math/Constant.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

#include <tuple>
#include <unordered_map>
#include <unordered_set>

using namespace mlir;

namespace onnx_mlir {

static Value rewriteMatMul(ONNXConstantOp COp, Value C1, Value C2,
    MultiDialectBuilder<OnnxBuilder> create, int64_t m, int64_t n, int64_t r) {
  Operation *useOp = *COp->getUsers().begin();
  if (useOp->getOperand(0).getDefiningOp() == COp) {
    // C is the first input
    // C1<m,r>xC2<r,n>xB<n, k>
    // C1<m,r>x(C2<r,n>xB<n, k>)

    Value B = useOp->getOperand(1);
    int64_t k = getShape(B.getType())[1];
    Type elementType = getElementType(B.getType());
    RankedTensorType tyR1 = RankedTensorType::get({r, k}, elementType);
    auto firstMatMul = create.onnx.matmul(tyR1, C2, B);
    RankedTensorType tyR2 = RankedTensorType::get({m, k}, elementType);
    auto secondMatMul = create.onnx.matmul(tyR2, C1, firstMatMul);
    return secondMatMul;
  } else {
    // C is the second input
    // A<k,m>xC1<m,r>xC2<r,n>
    // (A<k,m>xC1<m,r>)xC2<r,n>
    Value A = useOp->getOperand(0);
    int64_t k = getShape(A.getType())[0];
    Type elementType = getElementType(A.getType());
    RankedTensorType tyR1 = RankedTensorType::get({k, r}, elementType);
    auto firstMatMul = create.onnx.matmul(tyR1, A, C1);
    RankedTensorType tyR2 = RankedTensorType::get({k, n}, elementType);
    auto secondMatMul = create.onnx.matmul(tyR2, firstMatMul, C2);
    secondMatMul.dump();
    return secondMatMul;
  }
};

LogicalResult MatrixDecomposePattern::matchAndRewrite(
    ONNXConstantOp constantOp, PatternRewriter &rewriter) const {
  // Assume that constantOp is a tensor<MxNxT>
  // It is decomposed into two constants:
  // -  C1 of tensor<MxRxT>
  // -  C2 of tensor<RxNxT>
  // The value of R, as well as the values of C1 and C2 should be determined
  // before the transformation.
  // The values of a constant could be put into a file.
  // There could be a summary file to provide the file names and value of R
  // for each constant to be decomposed.
  // Here in the experiment, it is assumed that only one constant to be
  // decomposed and all the relavent info is hard-wired in compiler.

  Type ty = constantOp->getResultTypes()[0];
  ArrayRef<int64_t> originConstantShape = getShape(ty);
  Type elementType = getElementType(ty);
  int64_t m = originConstantShape[0];
  int64_t n = originConstantShape[1];
  int64_t r = 2; // the value is chosen randomly

  RankedTensorType tyC1 = RankedTensorType::get({m, r}, elementType);
  RankedTensorType tyC2 = RankedTensorType::get({r, n}, elementType);
  ValueRange inputs = constantOp->getOperands();

  Value C1, C2;
  if (stage == 1) { // stage one
    StringAttr funcNameAttr = rewriter.getStringAttr("DecompseConstant");
    std::vector<Type> outputTypes({tyC1, tyC2});
    // Create a new constant just with different location
    Value newC =
        rewriter.create<ONNXConstantOp>(UnknownLoc::get(rewriter.getContext()),
            Attribute(), constantOp.getValueAttr());
    ONNXCustomOp COp =
        rewriter.create<ONNXCustomOp>(constantOp->getLoc(), outputTypes, newC);
    COp->setAttr("function_name", funcNameAttr);
    C1 = COp.getResults()[0];
    C2 = COp.getResults()[1];
  } else { // stage two
    StringAttr funcNameAttr = rewriter.getStringAttr("getConstant");
    ONNXCustomOp C1Op =
        rewriter.create<ONNXCustomOp>(constantOp->getLoc(), tyC1, inputs);
    ONNXCustomOp C2Op =
        rewriter.create<ONNXCustomOp>(constantOp->getLoc(), tyC2, inputs);
    C1Op->setAttr("function_name", funcNameAttr);

    // Filename is determined by the constant loc
    Location loc = constantOp->getLoc();
    NameLoc nameLoc = loc.cast<NameLoc>();
    llvm::StringRef name = nameLoc.getName().getValue();
    StringAttr fileNameAttr1 = rewriter.getStringAttr(name + "_1.txt");
    C1Op->setAttr("file_name", fileNameAttr1);

    C2Op->setAttr("function_name", funcNameAttr);
    StringAttr fileNameAttr2 = rewriter.getStringAttr(name + "_2.txt");
    C2Op->setAttr("file_name", fileNameAttr2);
    C1 = C1Op.getResults()[0];
    C2 = C2Op.getResults()[0];
  }

  // Check the use of the constant, assuming only one user
  Operation *useOp = *constantOp->getUsers().begin();
  MultiDialectBuilder<OnnxBuilder> create(rewriter, useOp->getLoc());
  Value resultVal;
  if (isa<ONNXMatMulOp>(useOp)) {
    resultVal = rewriteMatMul(constantOp, C1, C2, create, m, n, r);
  } // else if (useOp.isa<ONNXGemmOp>()) {
    // resultVal = rewriteGemm(constantOp, C1, C2, creat);
  //}
  else {
    llvm_unreachable("expected");
  }
  rewriter.replaceOp(useOp, resultVal);

  rewriter.eraseOp(constantOp);
  return success();
}

bool MatrixDecomposePattern::toDecompose(ONNXConstantOp constantOp,
    MatrixDecomposeVectorType constantList, int stage) {
  static const int64_t SIZE_THRESHOLD = 2;

  // Check the possible candidate

  // Check the shape
  Type ty = constantOp->getResultTypes()[0];
  int64_t rank = getRank(ty);
  if (rank != 2)
    return false;

  ArrayRef<int64_t> shape = getShape(ty);
  for (int64_t dim : shape) {
    if (dim < SIZE_THRESHOLD)
      return false;
  }
  // Check the usage of the constant.
  if (!constantOp->hasOneUse())
    return false;
  Operation *useOp = *constantOp->getUsers().begin();
  if (!isa<ONNXMatMulOp, ONNXGemmOp>(useOp))
    return false;

  Location loc = constantOp->getLoc();
  if (loc.isa<NameLoc>()) {
    // ToFix: Use location.walker to handle fused Location
    NameLoc nameLoc = loc.cast<NameLoc>();
    llvm::StringRef name = nameLoc.getName().getValue();
    if (stage == 0) {
      // Scanning mode: print out all the candidate
      printf(
          "Candiate constant %s %lldx%lld\n", name.data(), shape[0], shape[1]);
      return false;
    } else {
      for (std::string specified : constantList) {
        if (specified == name) {
          return true;
        }
      }
      return false;
    }
  }

  return true;
}

} // namespace onnx_mlir
