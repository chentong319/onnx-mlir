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
  FILE *fp = fopen("hisim_model.csv", "w");
  getOperation()->walk([&](HISIMComputationOpInterface hisimOp) {
    //OpBuilder builder(getOperation()->getContext());
    //builder.setInsertionPoint(verifiableOp);
    std::vector<int> res = hisimOp.HISIMComputation();
    for (unsigned long  index = 0; index < res.size()-1; index++) {
      fprintf(fp, "%d,", res[index]);
    }
    fprintf(fp, "%d\n", res[res.size()-1]);
  });
  fclose(fp);
}

std::unique_ptr<Pass> onnx_mlir::createGenerateHISIMComputationPass() {
  return std::make_unique<GenerateHISIMComputationPass>();
}
