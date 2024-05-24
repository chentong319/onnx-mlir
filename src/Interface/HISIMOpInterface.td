// SPDX-License-Identifier: Apache-2.0

//===- HISIMOpInterface.td - HISIMOp Interface -*- tablegen -===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// Defines the operations of HISIM Op Interface.
//
//===----------------------------------------------------------------------===//

#ifdef HISIME_OP_INTERFACE
#else
#define HISIME_OP_INTERFACE

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def HISIMOpOpInterface : OpInterface<"HISIMOpOpInterface"> {
  let description = [{
    Interface to create the information of this Op which is needed by HISIM
    simulator. For example, the computation and/or communication requirement
    of this op. Such information could be used to generate the model description
    for HISIM simulator.
  }];

  let methods = [
    InterfaceMethod<[{Return computation workload of this Op.
                      The default implement is for examplary and debugging
                      purpose, and copied for ResNet110. Each op should
                      implement the interface according to its operation
                      and input size}],
                    "std::vector<int>", "HISIMComputation", (ins),
                    /*methodBody=*/[{ return {32,32,16,3,3,16,0,1,1}; 
    }] >
  ];
  
  let cppNamespace = "::mlir";
}

#endif // HISIME_OP_INTERFACE
