#include "mlir/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#include "src/Interface/HISIMOpInterface.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_GENERATEHISIMCOMPUTATION
#include "src/Transform/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct GenerateHISIMComputationPass
    : public impl::GenerateHISIMComputationBase<
          GenerateHISIMComputationPass> {
  void runOnOperation() override;
};
} // namespace

void GenerateHISIMComputationPass::runOnOperation() {
  getOperation()->walk([&](HISIMComputationOpInterface hisimOp) {
    //OpBuilder builder(getOperation()->getContext());
    //builder.setInsertionPoint(verifiableOp);
    hisimOp.HISIMComputation();
  });
}

std::unique_ptr<Pass> onnx_mlir::createGenerateHISIMComputationPass() {
  return std::make_unique<GenerateHISIMComputationPass>();
}
