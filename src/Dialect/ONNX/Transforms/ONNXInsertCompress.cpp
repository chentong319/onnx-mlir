/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ONNXInsertCompress.cpp - ONNX high level transformation --------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This pass is to set onnx_node_name attribute for ONNX operations if the
// attribute is absent.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "onnx-insert-compress"

using namespace mlir;
using namespace onnx_mlir;

namespace {

struct ONNXInsertCompressPass
    : public PassWrapper<ONNXInsertCompressPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXInsertCompressPass)

  StringRef getArgument() const override { return "onnx-insert-compress"; }

  StringRef getDescription() const override {
    return "Insert compression op for ONNX operations if the data type" 
           "is F32 or F64";
  }

  void runOnOperation() final;

};

void ONNXInsertCompressPass::runOnOperation() {
  //ModuleOp moduleOp = getOperation();
  //MLIRContext *context = &getContext();

  getOperation().walk([&](Operation *op) -> WalkResult {
    // Only deal with ONNX ops.
    if (op->getDialect()->getNamespace() != ONNXDialect::getDialectNamespace())
      return WalkResult::advance();

    // Ops to skip
    if (isa<ONNXCustomOp, ONNXReturnOp>(op)) {
      return WalkResult::advance();
    }

    //if (op->getResults().size() != 1)
      //return WalkResult::advance();


    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);
    // Compress the input, if not compressed
    
    // Compress the output
    for(Value result :  op->getResults()) {
      if (getElementType(result.getType()).isa<FloatType>()) {
        ONNXCustomOp compressOp = builder.create<ONNXCustomOp>(op->getLoc(), result.getType(), result);
        StringAttr funcNameAttr = builder.getStringAttr("omCustomizedCompressFloat");
        compressOp->setAttr("function_name", funcNameAttr);
        StringAttr shapeAttr = builder.getStringAttr("SameAs");
        compressOp->setAttr("shape_infer_pattern", shapeAttr);
        result.replaceAllUsesExcept(compressOp.getResult(0), compressOp);
     }
    }
    return WalkResult::advance();
  });
}

} // namespace

namespace onnx_mlir {

/*!
 * Create a ONNXInsertCompress pass.
 */
std::unique_ptr<mlir::Pass> createONNXInsertCompressPass() {
  return std::make_unique<ONNXInsertCompressPass>();
}

} // namespace onnx_mlir
