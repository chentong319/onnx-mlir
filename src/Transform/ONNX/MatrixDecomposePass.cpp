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

#include <fstream>

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

  Option<std::string> onnxMatrixDecomposeFile{*this,
      "onnx-matrix-decompose-file",
      llvm::cl::desc(
          "name of file that specify which constant to be decomposed."),
      llvm::cl::init("matrix_decompose.txt")};

  Option<int> onnxMatrixDecomposeStage{*this, "onnx-matrix-decompose-stage",
      llvm::cl::desc("which stage for constant decomposition."
                     "0: list candidate"
                     "1: online decomposition(default)"
                     "2: offline decomposition"),
      llvm::cl::init(1)};

  MatrixDecomposePass() = default;
  MatrixDecomposePass(const MatrixDecomposePass &pass)
      : mlir::PassWrapper<MatrixDecomposePass, OperationPass<func::FuncOp>>() {
    this->onnxMatrixDecomposeFile = pass.onnxMatrixDecomposeFile;
    this->onnxMatrixDecomposeStage = pass.onnxMatrixDecomposeStage;
    this->matrixToDecompose = pass.matrixToDecompose;
    this->stage = pass.stage;
  }

  MatrixDecomposePass(std::string fileName, int stage) {
    this->onnxMatrixDecomposeFile = fileName;
    this->stage = stage;
  }

  LogicalResult initialize(MLIRContext *context) override {
    stage = onnxMatrixDecomposeStage;
    // Read the file
    matrixToDecompose.clear();
    std::ifstream inFile;
    inFile.open(onnxMatrixDecomposeFile);
    if (inFile) {
      std::string locName;
      while (inFile >> locName) {
        // printf("string %s\n", locName.c_str());
        matrixToDecompose.push_back(locName);
      }
      inFile.close();
    } else {
      // Do nothing
      // The pass is used to file out all the candidate;
    }

    return success();
  }

  void runOnOperation() final {
    func::FuncOp f = getOperation();

    ConversionTarget target(getContext());
    // This sentence is needed.
    // Not clear about its usage.
    target
        .addLegalDialect<ONNXDialect, arith::ArithDialect, func::FuncDialect>();

    target.addDynamicallyLegalOp<ONNXConstantOp>([this](ONNXConstantOp op) {
      return !onnx_mlir::MatrixDecomposePattern::toDecompose(
          op, matrixToDecompose, stage);
    });

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<onnx_mlir::MatrixDecomposePattern>(
        context, matrixToDecompose, stage);
    if (failed(applyPartialConversion(f, target, std::move(patterns))))
      signalPassFailure();
  }

  // Data to control matrix decompose
  onnx_mlir::MatrixDecomposeVectorType matrixToDecompose;
  int stage;
};

} // end anonymous namespace

/*!
 * Create an matrix decompose pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createMatrixDecomposePass() {
  return std::make_unique<MatrixDecomposePass>();
}

std::unique_ptr<mlir::Pass> onnx_mlir::createMatrixDecomposePass(
    std::string fileName, int stage) {
  return std::make_unique<MatrixDecomposePass>(fileName, stage);
}
