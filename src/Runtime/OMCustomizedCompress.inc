
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

#define DEFAULT_SCHEME 0

void omCustomizedCompressFloat(OMTensor *outputTensor, OMTensor *inputTensor, char *op_name) {
  const OM_DATA_TYPE dataType = omTensorGetDataType(inputTensor);
  static int scheme = DEFAULT_SCHEME;
  static bool isFirst = true;
  if (isFirst == true) {
    char *scheme_str;
    if ((scheme_str = getenv("COMPRESS_SCHEME"))) {
      if (strncmp(scheme_str, "1", 1) == 0){
        scheme = 1;
      } else if (strncmp(scheme_str, "2", 1) == 0){
        scheme = 2;
      }
    }
    isFirst = false;
    printf("customized compress scheme %d\n", scheme);
  }

  if (getenv("COMPRESS_DEBUG")) {
    printf("customized press op %s\n", op_name);
  }
  
  if (dataType == ONNX_TYPE_FLOAT) {
    float *inputPtr = (float *)omTensorGetDataPtr(inputTensor);
    float *outputPtr = (float *)omTensorGetDataPtr(outputTensor);
    for(uint i = 0; i < omTensorGetNumElems(inputTensor); i ++) {
      unsigned int *pi = (unsigned int*) &inputPtr[i];
      unsigned int  s = *pi >> 31;
      unsigned int  e = *pi & 0x7f800000;
      e >>= 23;
      unsigned int  m = *pi & 0x007fffff;


      if (scheme == 0) {
    // No modification and just copy the input

      } else if (scheme == 1) {
        // Only drop bits in matissa
        m &= 0x00700000;
      } else if (scheme == 2) {
        // Keep 3 bits for matissa and 5 bits for exponent
        // Clip the exponent to 127-16 to 127+15
#define MYE 16
        m &= 0x00700000;
        if (e < 127-MYE) e = 127-MYE;
        if (e > 127+MYE) e = 127+MYE-1;
      } else {
        assert(0 && "invalid scheme");
      }
      // Write to output
      unsigned int *po = (unsigned int *) outputPtr+i;
      *po = (s<<31) | (e<<23) | m;
    } // end loop

// CHECK correctness
/*
    for(uint i = 0; i < omTensorGetNumElems(inputTensor); i ++) {
      printf("%f %f\n", inputPtr[i], outputPtr[i]);
    }
*/
  } else {
    assert(0 && "unsupported type");
  }

  return;
}

