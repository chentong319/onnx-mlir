
#ifdef __cplusplus
#include <cassert>
#else
#include <assert.h>
#endif

#ifndef __USE_GNU
#define __USE_GNU
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "onnx-mlir/Runtime/OMTensor.h"
#include "onnx-mlir/Runtime/OnnxDataType.h"
#ifdef __cplusplus
#include "src/Runtime/OMTensorHelper.hpp"
#endif

#define COMPRESS_SCHEME 0

void omCustomizedCompressFloat(OMTensor *outputTensor, OMTensor *inputTensor) {
  const OM_DATA_TYPE dataType = omTensorGetDataType(inputTensor);

  if (dataType == ONNX_TYPE_FLOAT) {
   float *inputPtr = (float *)omTensorGetDataPtr(inputTensor);
    float *outputPtr = (float *)omTensorGetDataPtr(outputTensor);
    for(uint i = 0; i < omTensorGetNumElems(inputTensor); i ++) {
      unsigned int *pi = (unsigned int*) &inputPtr[i];
      unsigned int  s = *pi >> 31;
      unsigned int  e = *pi & 0x7f800000;
      e >>= 23;
      unsigned int  m = *pi & 0x007fffff;


#if (COMPRESS_SCHEME == 0)
  // No modification and just copy the input

#elif (COMPRESS_SCHEME == 1)
// Only drop bits in matissa
    m &= 0x00700000;
#elif (COMPRESS_SCHEME == 2)
// Keep 3 bits for matissa and 5 bits for exponent
// Clip the exponent to 127-16 to 127+15
#define MYE 16
      m &= 0x00700000;
      if (e < 127-MYE) e = 127-MYE;
      if (e > 127+MYE) e = 127+MYE-1;

#else
#error COMPRESS_SCHEME is not defined or handled
#endif
// Write to output
      outputPtr[i] = (s<<31) | (e<<23) | m;
} // end loop
  } else {
    assert(0 && "unsupported type");
  }
  return;
}

