#pragma once

#include "../../src/vent.h"
#include "../../src/Sciara.h"


__global__ void computeOutflows_Tiled_wH(
            double *sh, double *st, double *sz, double *mf);


__global__ void massBalance_Tiled_wH(
    double *sh, double *sh_next, double *st, double *st_next, double *mf);  


__global__ void computeNewTemperatureAndSolidification_Tiled_wH();

__global__ void boundaryConditions_Tiled_wH();