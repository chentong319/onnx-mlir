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

class MatrixDecomposePattern
    : public mlir::OpRewritePattern<mlir::ONNXConstantOp> {
public:
  using mlir::OpRewritePattern<mlir::ONNXConstantOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXConstantOp ConstantOp,
      mlir::PatternRewriter &rewriter) const override;

  static bool toDecompose(mlir::ONNXConstantOp, std::string flag);
};

} // namespace onnx_mlir
