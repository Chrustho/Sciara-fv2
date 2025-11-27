#include <cuda_runtime.h>
#include "../../src/Sciara.h"

void allocateSubstates_proj(Sciara *sciara)
{
    size_t size_double = sciara->domain->rows * sciara->domain->cols * sizeof(double);
    size_t size_bool = sciara->domain->rows * sciara->domain->cols * sizeof(bool);
    size_t size_Mf = size_double * NUMBER_OF_OUTFLOWS;

    cudaMallocManaged((void **)&sciara->substates->Sz, size_double);
    cudaMallocManaged((void **)&sciara->substates->Sz_next, size_double);

    cudaMallocManaged((void **)&sciara->substates->Sh, size_double);
    cudaMallocManaged((void **)&sciara->substates->Sh_next, size_double);

    cudaMallocManaged((void **)&sciara->substates->ST, size_double);
    cudaMallocManaged((void **)&sciara->substates->ST_next, size_double);

    cudaMallocManaged((void **)&sciara->substates->Mf, size_Mf);
    cudaMallocManaged((void **)&sciara->substates->Mb, size_bool);
    cudaMallocManaged((void **)&sciara->substates->Mhs, size_double);

    cudaDeviceSynchronize();
}

void deallocateSubstates(Sciara *sciara)
{
    if (sciara->substates->Sz)
        cudaFree(sciara->substates->Sz);
    if (sciara->substates->Sz_next)
        cudaFree(sciara->substates->Sz_next);

    if (sciara->substates->Sh)
        cudaFree(sciara->substates->Sh);
    if (sciara->substates->Sh_next)
        cudaFree(sciara->substates->Sh_next);

    if (sciara->substates->ST)
        cudaFree(sciara->substates->ST);
    if (sciara->substates->ST_next)
        cudaFree(sciara->substates->ST_next);

    if (sciara->substates->Mf)
        cudaFree(sciara->substates->Mf);
    if (sciara->substates->Mb)
        cudaFree(sciara->substates->Mb);
    if (sciara->substates->Mhs)
        cudaFree(sciara->substates->Mhs);

    cudaDeviceSynchronize();
}
