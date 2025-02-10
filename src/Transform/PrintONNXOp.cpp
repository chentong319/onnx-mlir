#include "mlir/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#include "src/Pass/Passes.hpp"

namespace mlir {
#define GEN_PASS_DEF_GENERATEHISIMCOMPUTATION
#include "src/Transform/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct PrintONNXOpPass : public impl::PrintONNXOpBase<PrintONNOPPass> {
  void runOnOperation() override;
};

void PrintONNXOpPass::runOnOperation() {
}

std::unique_ptr<Pass> onnx_mlir::createPrintONNXOpPass() {
  return std::make_unique<GeneratePrintONNXOpPass>();
}
