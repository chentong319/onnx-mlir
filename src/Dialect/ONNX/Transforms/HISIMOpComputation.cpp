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
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

/*
IFM_Size_x:            Size of the input of the Layer in x-dimension
IFM_Size_y:            Size of the input of the Layer in y-dimension
N_IFM:                 Number of input channels of the layer   
Kx:                    Kernel size in x-dimension of the layer
Ky:                    Kernel size in y-dimension of the layer
NOFM:                  Number of output channels of the layer  
pool:                  Parameter indicating if the layer is followed by pooling or not: 0 if not followed by pooling and 1 if followed by pooling
layer-wise sparsity:   Total Sparsity of the layer
*/

struct ConvOpInterface
    : public HISIMComputationOpInterface::ExternalModel<ConvOpInterface, ONNXConvOp> {
  std::vector<int>  HISIMComputation(Operation *op) const {
    ONNXConvOp convOp = cast<ONNXConvOp>(op);

    /*
     Input data tensor from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 … x Dn).
    */
    auto X = convOp.getX();
    auto XType = X.getType();
    if (!onnx_mlir::hasStaticShape(XType)) {
      op->dump();
      return {32,32,16,3,3,16,0,1,1};
    }
    llvm::ArrayRef<int64_t> XShape = onnx_mlir::getShape(XType);
    if (XShape.size() != 4) {
      op->dump();
      return {32,32,16,3,3,16,0,1,1};
    }
    std::vector<int> res;
    
    res.push_back(XShape[2]); //IFM_Size_X
    res.push_back(XShape[3]); // IFM_Size_y
    res.push_back(XShape[1]); // N_IFM

    auto W = convOp.getW();
    auto WType = W.getType();
    if (!onnx_mlir::hasStaticShape(WType)) {
      op->dump();
      return {32,32,16,3,3,16,0,1,1};
    }
    llvm::ArrayRef<int64_t> WShape = onnx_mlir::getShape(WType);
    if (WShape.size() != 4) {
      op->dump();
      return {32,32,16,3,3,16,0,1,1};
    }
    res.push_back(WShape[2]); // Kx
    res.push_back(WShape[3]); // Ky
    res.push_back(WShape[0]); // NOFM

    // Followed by Pool
    // ToFix: the real model is followed by activation function and then Pool
    /* 
    %428 = "onnx.Conv"(%arg0, %426, %427) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [7, 7], onnx_node_name = "/features/conv0/Conv", pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %429 = "onnx.Relu"(%428) {onnx_node_name = "/features/relu0/Relu"} : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %430 = "onnx.MaxPoolSingleOut"(%429) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], onnx_node_name = "/features/pool0/MaxPool", pads = [1, 1, 1, 1], storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32>
    */
    res.push_back(1);

    // Sparsity
    res.push_back(1);

    return res;
  }
};

std::vector<int> elementwiseHISIMComputation(Operation *op) {
  auto X = op->getOperands()[0];
  auto XType = X.getType();
  if (!onnx_mlir::hasStaticShape(XType)) {
    op->dump();
    return {32,32,16,3,3,16,0,1,1};
  }
  llvm::ArrayRef<int64_t> XShape = onnx_mlir::getShape(XType);
  if (XShape.size() != 4) {
    op->dump();
    return {32,32,16,3,3,16,0,1,1};
  }
  std::vector<int> res;
    
  res.push_back(XShape[2]); //IFM_Size_X
  res.push_back(XShape[3]); // IFM_Size_y
  res.push_back(XShape[1]); // N_IFM

  res.push_back(1); // Kx
  res.push_back(1); // Ky
  res.push_back(XShape[1]); // NOFM

  res.push_back(1);

  // Sparsity
  res.push_back(1);

  return res;
}

struct AddOpInterface
    : public HISIMComputationOpInterface::ExternalModel<AddOpInterface, ONNXAddOp> {
  std::vector<int>  HISIMComputation(Operation *op) const {
    return elementwiseHISIMComputation(op);
  }
};

struct MaxPoolOpInterface
    : public HISIMComputationOpInterface::ExternalModel<MaxPoolOpInterface, ONNXMaxPoolOp> {
  std::vector<int>  HISIMComputation(Operation *op) const {
    return elementwiseHISIMComputation(op);
  }
};

struct MaxPoolSingleOutOpInterface
    : public HISIMComputationOpInterface::ExternalModel<MaxPoolSingleOutOpInterface, ONNXMaxPoolSingleOutOp> {
  std::vector<int>  HISIMComputation(Operation *op) const {
    return elementwiseHISIMComputation(op);
  }
};

struct ReluOpInterface
    : public HISIMComputationOpInterface::ExternalModel<ReluOpInterface, ONNXReluOp> {
  std::vector<int>  HISIMComputation(Operation *op) const {
    return elementwiseHISIMComputation(op);
  }
};

void mlir::registerHISIMComputationOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, mlir::ONNXDialect *dialect) {
    ONNXConvOp::attachInterface<ConvOpInterface>(*ctx);
    ONNXReluOp::attachInterface<ReluOpInterface>(*ctx);
    //ONNXAddOp::attachInterface<AddOpInterface>(*ctx);
    ONNXMaxPoolOp::attachInterface<MaxPoolOpInterface>(*ctx);
    ONNXMaxPoolSingleOutOp::attachInterface<MaxPoolSingleOutOpInterface>(*ctx);
        // Load additional dialects of which ops may get created.
    ctx->loadDialect<ONNXDialect>();
  });    
}   
