
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

#define DEFAULT_SCHEME 1
#define DEFAULT_MBITS 0
#define DEFAULT_EBITS 0

static unsigned int  compress_matissa(unsigned int m, int nbits) {
  if (nbits == 0)
    return m;

  unsigned int mask=1;
  for (int i=1; i < nbits ; i++) {
    mask = mask<<1 | 0x1 ;
  }

  mask = mask << (23-nbits);
  //printf("%u\n", mask);

  unsigned newm = m & mask; // get first nbits.
  newm = (0x1 << (23-nbits-1)) | newm; // attach a 1 to reduce the roundup error
  return newm;
}

static unsigned int compress_exponent(unsigned int e, int nbits) {
#define EBITS 8
  if (nbits == 0)
    return e;
  if (nbits == 1 || nbits > EBITS)
    assert(0 && "nbit value is incorrect");

  unsigned int half = 1;
  half = half << (EBITS-1);
  half -= 1;

  unsigned clipbound = (1<< (nbits-1));
  unsigned int newe = e;
  if (e > half + clipbound)
    newe = half+clipbound-1;
  else if (e < half - clipbound)
    newe = half-clipbound;

  return newe;
}


void omCustomizedCompressFloat(OMTensor *outputTensor, OMTensor *inputTensor, char *op_name) {
  const OM_DATA_TYPE dataType = omTensorGetDataType(inputTensor);
  static bool isFirst = true;
  static int scheme = DEFAULT_SCHEME;
  static int mbits = DEFAULT_MBITS;
  static int ebits = DEFAULT_EBITS;
  if (isFirst == true) {
    char *scheme_str;
    if ((scheme_str = getenv("COMPRESS_SCHEME"))) {
      sscanf(scheme_str, "%d", &scheme);
    }
    char *mbits_str;
    if ((mbits_str = getenv("COMPRESS_MBITS"))) {
      sscanf(mbits_str, "%d", &mbits);
    }
    char *ebits_str;
    if ((ebits_str = getenv("COMPRESS_EBITS"))) {
      sscanf(ebits_str, "%d", &ebits);
    }
    isFirst = false;
    printf("customized compress config: scheme %d, mbits %d ebits %d\n", scheme, mbits, ebits);
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
        m = compress_matissa(m, mbits);
        e = compress_exponent(e, ebits);
      } else if (scheme == 2) {
        // Keep 3 bits for matissa and 5 bits for exponent
        // Clip the exponent to 127-16 to 127+15
#define MYE 16
        m = compress_matissa(m, mbits);
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

