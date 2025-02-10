#ifndef ONNX_MLIR_TRANSFORMS_PASSES
#define ONNX_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def PrintONNXOp : Pass<"print-onnx-op"> {
  let summary = "Print the input/output tensors of selected ONNX op";
  let description = [{
    This pass inserts krnl.print after the onnx op to print the value of the 
    input/output tensors at runtime.
  }];
  let constructor = "mlir::createPrintONNXOpPass()";
}

#endif
