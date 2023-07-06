/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- DecomposeEinsum.cpp - Decompose Einsum op ----------------===//
//
// This file implements the decomposition of ONNXEinsumOp to simpler ops.
//
//===----------------------------------------------------------------------===//

#include "src/Transform/ONNX/MatrixDecompose.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
//#include "src/Dialect/ONNX/ONNXOps/Math/Constant.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

#include <tuple>
#include <unordered_map>
#include <unordered_set>

using namespace mlir;

namespace onnx_mlir {

static Value rewriteMatMul(ONNXConstantOp C, ONNXCustomOp C1, ONNXCustomOp C2, MultiDialectBuilder<OnnxBuilder> create, int64_t m, int64_t n, int64_t r) {
  Operation *useOp = *C->getUsers().begin();
  if (useOp->getOperand(0).getDefiningOp() == C) {
    // C is the first input
    // C1<m,r>xC2<r,n>xB<n, k>
    // C1<m,r>x(C2<r,n>xB<n, k>)
     
    Value B = useOp->getOperand(1);
    int64_t k = getShape(B.getType())[1];
    Type elementType = getElementType(B.getType());
    RankedTensorType tyR1 = RankedTensorType::get({r, k}, elementType);
    auto firstMatMul = create.onnx.matmul(tyR1, C2.getResults()[0], B);
    RankedTensorType tyR2 = RankedTensorType::get({m, k}, elementType);
    auto secondMatMul = create.onnx.matmul(tyR2, C1.getResults()[0], firstMatMul);
    return secondMatMul;
  } else {
    // C is the second input
    // A<k,m>xC1<m,r>xC2<r,n>
    // (A<k,m>xC1<m,r>)xC2<r,n>
    Value A = useOp->getOperand(0);
    int64_t k = getShape(A.getType())[0];
    Type elementType = getElementType(A.getType());
    RankedTensorType tyR1 = RankedTensorType::get({k, r}, elementType);
    auto firstMatMul = create.onnx.matmul(tyR1, A, C1.getResults()[0]);
    RankedTensorType tyR2 = RankedTensorType::get({k, n}, elementType);
    auto secondMatMul = create.onnx.matmul(tyR2, firstMatMul, C2.getResults()[0]);
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
  int64_t r = 2;  // the value is chosen randomly

  RankedTensorType tyC1 = RankedTensorType::get({m,r}, elementType);
  RankedTensorType tyC2 = RankedTensorType::get({r,n}, elementType);

  StringAttr funcNameAttr = rewriter.getStringAttr("getConstant");
  ValueRange inputs = constantOp->getOperands();
  
  ONNXCustomOp C1Op = rewriter.create<ONNXCustomOp>(constantOp->getLoc(), tyC1, inputs);
  ONNXCustomOp C2Op = rewriter.create<ONNXCustomOp>(constantOp->getLoc(), tyC2, inputs);
  C1Op->setAttr("function_name", funcNameAttr);
  StringAttr fileNameAttr1 = rewriter.getStringAttr("c1.txt");
  C1Op->setAttr("file_name", fileNameAttr1);

  C2Op->setAttr("function_name", funcNameAttr);
  StringAttr fileNameAttr2 = rewriter.getStringAttr("c2.txt");
  C2Op->setAttr("file_name", fileNameAttr2);

  // Check the use of the constant, assuming only one user
  Operation *useOp = *constantOp->getUsers().begin();
  MultiDialectBuilder<OnnxBuilder> create(rewriter, useOp->getLoc());
  Value resultVal;
  if (isa<ONNXMatMulOp>(useOp)) {
    resultVal = rewriteMatMul(constantOp, C1Op, C2Op, create, m, n, r);
  } //else if (useOp.isa<ONNXGemmOp>()) {
    //resultVal = rewriteGemm(constantOp, C1Op, C2Op, creat);
  //}
  else {
    llvm_unreachable("expected");
  }
  rewriter.replaceOp(useOp, resultVal);
 
  rewriter.eraseOp(constantOp);
  return success();
}

bool MatrixDecomposePattern::toDecompose(ONNXConstantOp constantOp, std::string flag) {
  // flag is intended to be used to compare label info.

  // Check the usage of the constant.
  if (!constantOp->hasOneUse())
    return false;
  Operation *useOp = *constantOp->getUsers().begin();
  if (!isa<ONNXMatMulOp, ONNXGemmOp>(useOp))
    return false;

  // Use the attribute as mark
  //if (!useOp->hasAttr("LowRankDecompose"))
    //return false;
  return true;
}

} // namespace onnx_mlir
