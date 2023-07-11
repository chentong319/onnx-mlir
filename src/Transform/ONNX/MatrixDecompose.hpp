/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- MatrixDecompose.hpp - Decompose Constant op ----------------===//
//
// This file implements the low rank decomposition of ONNXConstant
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/PatternMatch.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace onnx_mlir {

typedef std::string MatrixDecomposeEntryType;

typedef std::vector<MatrixDecomposeEntryType> MatrixDecomposeVectorType;

class MatrixDecomposePattern
    : public mlir::OpRewritePattern<mlir::ONNXConstantOp> {
public:
  using mlir::OpRewritePattern<mlir::ONNXConstantOp>::OpRewritePattern;

  MatrixDecomposePattern(
      mlir::MLIRContext *context, MatrixDecomposeVectorType table)
      : OpRewritePattern<mlir::ONNXConstantOp>(context),
        matrixToDecompose(table){};

  mlir::LogicalResult matchAndRewrite(mlir::ONNXConstantOp ConstantOp,
      mlir::PatternRewriter &rewriter) const override;

  static bool toDecompose(mlir::ONNXConstantOp, MatrixDecomposeVectorType list);

  MatrixDecomposeVectorType matrixToDecompose;
};

} // namespace onnx_mlir
