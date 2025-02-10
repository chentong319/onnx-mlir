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
    allowedOps.setRegexString(printONNXOpNodeName);
  getOperation().walk([&](mlir::Operation *op) -> WalkResult {
    // Check whether the current op is the specified one
    bool skip = false;
    if (printONNXOpNodeName != "")
      StringAttr nodeName =
          op->getAttrOfType<mlir::StringAttr>("onnx_node_name");
      if (nodeName && !nodeName.getValue().empty()) {
        skip = true;
      } else {
        skip = !nodeName.getValue().str().constains(printONNXOpNodeName);
      }
    }
    if (skip)
      return;
    
}

std::unique_ptr<Pass> onnx_mlir::createPrintONNXOpPass() {
  return std::make_unique<GeneratePrintONNXOpPass>();
}
