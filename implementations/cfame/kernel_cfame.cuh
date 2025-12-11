#pragma once

#include "../../src/vent.h"
#include "../../src/Sciara.h"

__global__ void CfAMe_Kernel(    const double* __restrict__ sh,
    const double* __restrict__ st,
    const double* __restrict__ sz,
    double* __restrict__ sh_next,
    double* __restrict__ st_next);