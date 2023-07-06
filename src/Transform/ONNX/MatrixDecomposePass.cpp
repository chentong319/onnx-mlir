/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- MatrixDecompsePass.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===----------------------------------------------------------------------===//

//#include "mlir/IR/OperationSupport.h"
//#include "mlir/Pass/PassManager.h"
//#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Transform/ONNX/MatrixDecompose.hpp"

using namespace mlir;

namespace {

struct MatrixDecompsePass : public mlir::PassWrapper<MatrixDecompsePass,
                                 OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatrixDecompsePass)

  StringRef getArgument() const override { return "onnx-matrix-decompose"; }

  StringRef getDescription() const override {
    return "Perform matrix decomposition and MatMul associativity rewriting.";
  }

  Option<std::string> onnxMatrixDecomposeFile{*this, "onnx-matrix-decompose-file",
      llvm::cl::desc("name of file that specify which constant to be decomposed."),
      llvm::cl::init("matrix_decompse.txt")};

  MatrixDecompsePass() = default;
  MatrixDecompsePass(const MatrixDecompsePass &pass)
      : mlir::PassWrapper<MatrixDecompsePass,
            OperationPass<func::FuncOp>>() {}

  MatrixDecompsePass(std::string fileName) {
    this->onnxMatrixDecomposeFile = fileName;
  }

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet cumulativePatterns(context);
    // Add patterns
    cumulativePatterns.insert<onnx_mlir::MatrixDecomposePattern>(context);
    patterns = FrozenRewritePatternSet(std::move(cumulativePatterns));
    return success();
  }
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    Region &body = f.getBody();

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    (void)applyPatternsAndFoldGreedily(body, patterns, config);
  }

  FrozenRewritePatternSet patterns;
}; 

} // end anonymous namespace

/*!
 * Create an matrix decompose pass.
 */
std::unique_ptr<mlir::Pass> createMatrixDecompsePass() {
  return std::make_unique<MatrixDecompsePass>();
}

std::unique_ptr<mlir::Pass> createMatrixDecompsePass(
    std::string fileName) {
  return std::make_unique<MatrixDecompsePass>(fileName);
}
