#ifndef ONNX_MLIR_TRANSFORMS_PASSES
#define ONNX_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def GenerateHISIMComputation : Pass<"generate-hisim-computation"> {
  let summary = "Generate computaton for HISIM";
  let description = [{ 
    This pass generates op-specific computation using
    `HISIMComputationOpInterface`. 
  }];
  let constructor = "mlir::createGenerateHISIMComputationPass()";
}   

#endif
