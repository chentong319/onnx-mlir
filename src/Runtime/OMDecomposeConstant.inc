/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- OMResize.cpp - OMTensor C/C++ Implementation ----------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains C/C++ neutral implementation of OMTensorList data
// structures and helper functions.
//
//===----------------------------------------------------------------------===//
#ifdef __cplusplus
#include <cassert>
#else
#include <assert.h>
#endif

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "onnx-mlir/Runtime/OMTensor.h"
#define R 4

// Get the data
float c1[][R] =
#include "input_1.txt"
;

float c2[R][32] = 
#include "input_2.txt"
;


void copyData(OMTensor *output, void *cptr) {
  int64_t rank = omTensorGetRank(output);
  if (rank != 2) {
    perror("Error: expect the rank is 2");
  }
  // Check the shape

  // Copy the data
  void *ptr1 = omTensorGetDataPtr(output);
  memcpy(ptr1, cptr, omTensorGetBufferSize(output));
}

void DecomposeConstant(
    OMTensor *output1, OMTensor *output2, OMTensor *input, int64_t rValue) {
  
    if (rValue != R)
      perror("Error: R value not match");
    copyData(output1, (void *) c1);
    copyData(output2, (void *) c2);
}
