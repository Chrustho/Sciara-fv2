#include "../../src/vent.h"
#include "../../src/Sciara.h"
#include "../../sciara_fv2.cpp"

void emitLava(
    int i,
    int j,
    Sciara *sciara) 
{
  for (int k = 0; k < sciara->simulation->vent.size(); k++)
  {
    if (i == sciara->simulation->vent[k].y() && j == sciara->simulation->vent[k].x())
    {
      double thickness_add = sciara->simulation->vent[k].thickness(sciara->simulation->elapsed_time, sciara->parameters->Pclock, sciara->simulation->emission_time, sciara->parameters->Pac);

      double current_Sh = GET(sciara->substates->Sh, sciara->domain->cols, i, j);

      SET(sciara->substates->Sh_next, sciara->domain->cols, i, j, current_Sh + thickness_add);
      
      SET(sciara->substates->ST_next, sciara->domain->cols, i, j, sciara->parameters->PTvent);

      sciara->simulation->total_emitted_lava += thickness_add;
    }
  }
}

__global__ void computeOutflows_Global(
    int r,
    int c,
    const int *Xi, 
    const int *Xj,
    const double *Sz,
    const double *Sh,
    const double *ST,
    double *Mf, 
    double Pc,
    double _a,
    double _b,
    double _c,
    double _d)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    int idx = i * c + j;

    double h0 = Sh[idx];
    if (h0 <= 0.0) return;

    bool eliminated[MOORE_NEIGHBORS];
    double z[MOORE_NEIGHBORS];
    double h[MOORE_NEIGHBORS];
    double H[MOORE_NEIGHBORS];
    double theta[MOORE_NEIGHBORS];
    double Pr[MOORE_NEIGHBORS]; 
    double w[MOORE_NEIGHBORS];
    
    double sz0 = Sz[idx];
    double T_val = ST[idx]; 

    double rr = pow(10.0, _a + _b * T_val);
    double hc = pow(10.0, _c + _d * T_val);

    for (int k = 0; k < MOORE_NEIGHBORS; k++)
    {

        int ni = i + Xi[k];
        int nj = j + Xj[k];
        int idx_k = ni * c + nj;
        if (idx_k < 0 || idx_k > r*c)
        {
            break;
        }
        

        double sz_k = Sz[idx_k];
        
        h[k] = Sh[idx_k]; 
        
        w[k] = Pc;
        Pr[k] = rr;

        if (k < VON_NEUMANN_NEIGHBORS)
            z[k] = sz_k;
        else
            z[k] = sz0 - (sz0 - sz_k) / sqrt(2.0); 
    }

    H[0] = z[0];
    theta[0] = 0.0;
    eliminated[0] = false;

    for (int k = 1; k < MOORE_NEIGHBORS; k++)
    {
        if (z[0] + h[0] > z[k] + h[k])
        {
            H[k] = z[k] + h[k];
            theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]);
            eliminated[k] = false;
        }
        else
        {
            eliminated[k] = true;
            H[k] = 0.0; 
            theta[k]=0.;
        }
    }

    bool loop;
    double avg;
    int counter;

    do
    {
        loop = false;
        avg = h[0];
        counter = 0;
        
        for (int k = 0; k < MOORE_NEIGHBORS; k++)
        {
            if (!eliminated[k])
            {
                avg += H[k];
                counter++;
            }
        }

        if (counter != 0)
            avg = avg / (double)counter;

        for (int k = 0; k < MOORE_NEIGHBORS; k++)
        {
            if (!eliminated[k] && avg <= H[k])
            {
                eliminated[k] = true;
                loop = true;
            }
        }
    } while (loop);

    for (int k = 1; k < MOORE_NEIGHBORS; k++)
    {
        int outflow_idx = k - 1; 
        int mf_idx = (outflow_idx * r * c) + idx;

        if (!eliminated[k] && h[0] > hc * cos(theta[k]))
        {
            Mf[mf_idx] = Pr[k] * (avg - H[k]);
        }
        else
        {
            Mf[mf_idx] = 0.0;
        }
    }
}


__global__ void massBalance_Global(
    int r,
    int c,
    const int *Xi,     
    const int *Xj,
    const double *Sh,       
    double *Sh_next,  
    const double *ST,       
    double *ST_next,  
    const double *Mf)       
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    int idx = i * c + j;

    const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3, 2, 1, 0, 6, 7, 4, 5};

    double initial_h = Sh[idx];
    double initial_t = ST[idx];
    
    double h_next = initial_h;
    double t_next = initial_h * initial_t; 

    int layer_size = r * c;

    for (int n = 1; n < MOORE_NEIGHBORS; n++)
    {
        int ni = i + Xi[n];
        int nj = j + Xj[n];
        int n_idx = ni * c + nj;

        int out_layer = n - 1;
        double outFlow = Mf[out_layer * layer_size + idx];

        int in_layer = inflowsIndices[n - 1];
        double inFlow = Mf[in_layer * layer_size + n_idx];

        double neigh_t = ST[n_idx];

        h_next += inFlow - outFlow;
        t_next += (inFlow * neigh_t - outFlow * initial_t);
    }

  
    if (h_next > 0.0)
    {
        t_next /= h_next;
        
        ST_next[idx] = t_next;
        Sh_next[idx] = h_next;
    }
}


__global__ void computeNewTemperatureAndSolidification_Global(
    int r, 
    int c,
    double Pepsilon, 
    double Psigma, 
    double Pclock, 
    double Pcool, 
    double Prho, 
    double Pcv, 
    double Pac, 
    double PTsol,
    const double *Sz,      
    double *Sz_next,       
    const double *Sh,      
    double *Sh_next,       
    const double *ST,      
    double *ST_next,       
    double *Mf,            
    double *Mhs,           
    const bool *Mb         
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    int idx = i * c + j;

    double h = Sh[idx];
    bool is_border = Mb[idx];

    if (h > 0.0 && !is_border)
    {
        double T = ST[idx];
        double z = Sz[idx];
        
        double numerator = 3.0 * pow(T, 3.0) * Pepsilon * Psigma * Pclock * Pcool;
        double denominator = Prho * Pcv * h * Pac;
        
        double aus = 1.0 + (numerator / denominator);

        double nT = T / pow(aus, 1.0/3.0);

        if (nT > PTsol) 
        {
            ST_next[idx] = nT;
        } 
        else 
        {   
            Sz_next[idx] = z + h;   
            Sh_next[idx] = 0.0;     
            ST_next[idx] = PTsol;   
            
            Mhs[idx] = Mhs[idx] + h;
        }
    }
}


__global__ void boundaryConditions_Global(
    int r,
    int c,
    const double *Mf,    
    const bool *Mb,      
    const double *Sh,    
    double *Sh_next,     
    const double *ST,    
    double *ST_next      
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    int idx = i * c + j;

    if (Mb[idx])
    {
        Sh_next[idx] = 0.0;
        ST_next[idx] = 0.0;
    }
}

__global__ void reduceAdd_Kernel(int n, const double *buffer, double *global_result)
{
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    double localSum = 0.0;
    while (i < n)
    {
        localSum += buffer[i];
        i += gridSize;
    }
    sdata[tid] = localSum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        double toAdd= sdata[0];
        atomicAdd(global_result, toAdd);
    }
}

/*
########## MAIN ############

int main(int argc, char **argv)
{
    Sciara *sciara;
    init(sciara);

    int max_steps = atoi(argv[MAX_STEPS_ID]);
    loadConfiguration(argv[INPUT_PATH_ID], sciara);

    allocateSubstates(sciara);

    int *d_Xi, *d_Xj;
    cudaMalloc((void**)&d_Xi, MOORE_NEIGHBORS * sizeof(int));
    cudaMalloc((void**)&d_Xj, MOORE_NEIGHBORS * sizeof(int));
    cudaMemcpy(d_Xi, sciara->X->Xi, MOORE_NEIGHBORS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Xj, sciara->X->Xj, MOORE_NEIGHBORS * sizeof(int), cudaMemcpyHostToDevice);

    int rows = sciara->domain->rows;
    int cols = sciara->domain->cols;
    
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    double total_current_lava = -1;
    simulationInitialize(sciara);

    util::Timer cl_timer;

    int reduceInterval = atoi(argv[REDUCE_INTERVL_ID]);
    double thickness_threshold = atof(argv[THICKNESS_THRESHOLD_ID]);

    size_t matrixSize = rows * cols * sizeof(double);

    while ((max_steps > 0 && sciara->simulation->step < max_steps) || 
           (sciara->simulation->elapsed_time <= sciara->simulation->effusion_duration) || 
           (total_current_lava == -1 || total_current_lava > thickness_threshold))
    {
        sciara->simulation->elapsed_time += sciara->parameters->Pclock;
        sciara->simulation->step++;

       
        cudaDeviceSynchronize();

        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                emitLava(i, j, rows, cols,
                         sciara->simulation->vent,
                         sciara->simulation->elapsed_time,
                         sciara->parameters->Pclock,
                         sciara->simulation->emission_time,
                         sciara->simulation->total_emitted_lava,
                         sciara->parameters->Pac,
                         sciara->parameters->PTvent,
                         sciara->substates->Sh,
                         sciara->substates->Sh_next,
                         sciara->substates->ST_next);
            }
        }


        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, matrixSize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, matrixSize, cudaMemcpyDeviceToDevice);


        computeOutflows_Global<<<grid, block>>>(
            rows, cols,
            d_Xi, d_Xj, // Usiamo i puntatori device allocati prima
            sciara->substates->Sz,
            sciara->substates->Sh,
            sciara->substates->ST,
            sciara->substates->Mf,
            sciara->parameters->Pc,
            sciara->parameters->a,
            sciara->parameters->b,
            sciara->parameters->c,
            sciara->parameters->d
        );


       
        massBalance_Global<<<grid, block>>>(
            rows, cols,
            d_Xi, d_Xj,
            sciara->substates->Sh,
            sciara->substates->Sh_next,
            sciara->substates->ST,
            sciara->substates->ST_next,
            sciara->substates->Mf
        );

        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, matrixSize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, matrixSize, cudaMemcpyDeviceToDevice);


     
        computeNewTemperatureAndSolidification_Global<<<grid, block>>>(
            rows, cols,
            sciara->parameters->Pepsilon,
            sciara->parameters->Psigma,
            sciara->parameters->Pclock,
            sciara->parameters->Pcool,
            sciara->parameters->Prho,
            sciara->parameters->Pcv,
            sciara->parameters->Pac,
            sciara->parameters->PTsol,
            sciara->substates->Sz,
            sciara->substates->Sz_next,
            sciara->substates->Sh,
            sciara->substates->Sh_next,
            sciara->substates->ST,
            sciara->substates->ST_next,
            sciara->substates->Mf,
            sciara->substates->Mhs,
            sciara->substates->Mb
        );

        cudaMemcpy(sciara->substates->Sz, sciara->substates->Sz_next, matrixSize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, matrixSize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, matrixSize, cudaMemcpyDeviceToDevice);


        boundaryConditions_Global<<<grid, block>>>(
            rows, cols,
            sciara->substates->Mf,
            sciara->substates->Mb,
            sciara->substates->Sh,
            sciara->substates->Sh_next,
            sciara->substates->ST,
            sciara->substates->ST_next
        );

        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, matrixSize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, matrixSize, cudaMemcpyDeviceToDevice);



        if (sciara->simulation->step % reduceInterval == 0)
        {
            total_current_lava = reduceAdd(rows, cols, sciara->substates->Sh);
        }
    }


    cudaDeviceSynchronize();
    
    double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
    printf("Step %d\n", sciara->simulation->step);
    printf("Elapsed time [s]: %lf\n", cl_time);
    printf("Emitted lava [m]: %lf\n", sciara->simulation->total_emitted_lava);
    printf("Current lava [m]: %lf\n", total_current_lava);

    printf("Saving output to %s...\n", argv[OUTPUT_PATH_ID]);
    saveConfiguration(argv[OUTPUT_PATH_ID], sciara);

    printf("Releasing memory...\n");
    cudaFree(d_Xi);
    cudaFree(d_Xj);
    deallocateSubstates(sciara); 
    finalize(sciara);

    return 0;
}
*/
