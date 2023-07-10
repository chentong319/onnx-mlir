/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- MatrixDecomposePass.cpp - ONNX Op Transform ------------------===//
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

struct MatrixDecomposePass : public mlir::PassWrapper<MatrixDecomposePass,
                                 OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatrixDecomposePass)

  StringRef getArgument() const override { return "onnx-matrix-decompose"; }

  StringRef getDescription() const override {
    return "Perform matrix decomposition and MatMul associativity rewriting.";
  }

  Option<std::string> onnxMatrixDecomposeFile{*this, "onnx-matrix-decompose-file",
      llvm::cl::desc("name of file that specify which constant to be decomposed."),
      llvm::cl::init("matrix_decompose.txt")};

  MatrixDecomposePass() = default;
  MatrixDecomposePass(const MatrixDecomposePass &pass)
      : mlir::PassWrapper<MatrixDecomposePass,
            OperationPass<func::FuncOp>>() {
    this->onnxMatrixDecomposeFile = pass.onnxMatrixDecomposeFile;
  }

  MatrixDecomposePass(std::string fileName) {
    this->onnxMatrixDecomposeFile = fileName;
  }

#if 0
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet cumulativePatterns(context);
    // Add patterns
    PatternBenefit highPriority(10000);
    cumulativePatterns.insert<onnx_mlir::MatrixDecomposePattern>(context, highPriority);
    patterns = FrozenRewritePatternSet(std::move(cumulativePatterns));
    return success();
  }
#endif

  void runOnOperation() final {
    func::FuncOp f = getOperation();
    //Region &body = f.getBody();

    //GreedyRewriteConfig config;
    //config.useTopDownTraversal = true;
    //(void)applyPatternsAndFoldGreedily(body, patterns, config);

    ConversionTarget target(getContext());
    target.addLegalDialect<ONNXDialect, arith::ArithDialect, func::FuncDialect>();

    target.addDynamicallyLegalOp<ONNXConstantOp>([](ONNXConstantOp op) {
      return !onnx_mlir::MatrixDecomposePattern::toDecompose(op, "");
    });

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<onnx_mlir::MatrixDecomposePattern>(context);
    if (failed(applyPartialConversion(f, target, std::move(patterns))))
      signalPassFailure();
  }

  //FrozenRewritePatternSet patterns;
}; 

} // end anonymous namespace

/*!
 * Create an matrix decompose pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createMatrixDecomposePass() {
  return std::make_unique<MatrixDecomposePass>();
}

std::unique_ptr<mlir::Pass> onnx_mlir::createMatrixDecomposePass(
    std::string fileName) {
  return std::make_unique<MatrixDecomposePass>(fileName);
}
