#include "src/Interface/HISIMOpInterface.hpp"
    
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h" 
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/HISIMOpComputation.hpp" 

using namespace mlir;

struct ConvOpInterface
    : public HISIMComputationOpInterface::ExternalModel<ConvOpInterface, ONNXConvOp> {
  std::vector<int>  HISIMComputation(Operation *op) const {
    ONNXConvOp convOp = cast<ONNXConvOp>(op);
    // ToFix
    op->dump();
    return {32,32,16,3,3,16,0,1,1};
  }
};

void mlir::registerHISIMComputationOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, mlir::ONNXDialect *dialect) {
    ONNXConvOp::attachInterface<ConvOpInterface>(*ctx);
        // Load additional dialects of which ops may get created.
    ctx->loadDialect<ONNXDialect, affine::AffineDialect, arith::ArithDialect,
                     cf::ControlFlowDialect>();
  });    
}   
