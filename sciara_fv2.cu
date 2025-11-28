#include "src/Sciara.h"
#include "src/io.h"
#include "src/util.hpp"
#include <cuda_runtime.h>
#include "implementations/global/kernel_global.cuh"

// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------
#define INPUT_PATH_ID          1
#define OUTPUT_PATH_ID         2
#define MAX_STEPS_ID           3
#define REDUCE_INTERVL_ID      4
#define THICKNESS_THRESHOLD_ID 5

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D indices
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------
void emitLava(
    int i,
    int j,
    int r,
    int c,
    vector<TVent> &vent,
    double elapsed_time,
    double Pclock,
    double emission_time,
    double &total_emitted_lava,
    double Pac,
    double PTvent,
    double *Sh,
    double *Sh_next,
    double *ST_next)
{
  for (int k = 0; k < vent.size(); k++)
    if (i == vent[k].y() && j == vent[k].x())
    {
      SET(Sh_next, c, i, j, GET(Sh, c, i, j) + vent[k].thickness(elapsed_time, Pclock, emission_time, Pac));
      SET(ST_next, c, i, j, PTvent);

      total_emitted_lava += vent[k].thickness(elapsed_time, Pclock, emission_time, Pac);
    }
}

void emitLava_global(
    int i,
    int j,
    Sciara *sciara) 
{
    // Parametri del dominio
    int rows= sciara->domain->rows;
    int cols= sciara->domain->cols;

    // Parametri della simulazione
    double pTvent= sciara->parameters->PTvent;
    double elapsed_time= sciara->simulation->elapsed_time;
    double pclock = sciara->parameters->Pclock;
    unsigned int em_time= sciara->simulation->emission_time; 
    double pac= sciara->parameters->Pac;
    double total_em_lava= sciara->simulation->total_emitted_lava;

    // Buffers
    double *sh=sciara->substates->Sh;
    double *sh_next= sciara->substates->Sh_next;
    double *st_next= sciara->substates->ST_next;

    int size= sciara->simulation->vent.size();

    for (int k = 0; k < size; k++)
    {
        TVent curr_vent= sciara->simulation->vent[k];

        if (i == curr_vent.y() && j == curr_vent.x())
    {

        double thickness_add = curr_vent.thickness(elapsed_time, pclock, em_time, pac);

        double current_Sh = GET(sh, cols, i, j);

        SET(sh_next, cols, i, j, current_Sh + thickness_add);

        SET(st_next, cols, i, j, pTvent);

        total_em_lava += thickness_add;
    }
    }
}

void computeOutflows(
    int i,
    int j,
    int r,
    int c,
    int *Xi,
    int *Xj,
    double *Sz,
    double *Sh,
    double *ST,
    double *Mf,
    double Pc,
    double _a,
    double _b,
    double _c,
    double _d)
{
  bool eliminated[MOORE_NEIGHBORS];
  double z[MOORE_NEIGHBORS];
  double h[MOORE_NEIGHBORS];
  double H[MOORE_NEIGHBORS];
  double theta[MOORE_NEIGHBORS];
  double w[MOORE_NEIGHBORS];  // Distances between central and adjecent cells
  double Pr[MOORE_NEIGHBORS]; // Relaiation rate arraj
  double f[MOORE_NEIGHBORS];
  bool loop;
  int counter;
  double sz0, sz, T, avg, rr, hc;

  if (GET(Sh, c, i, j) <= 0)
    return;

  T = GET(ST, c, i, j);
  rr = pow(10, _a + _b * T);
  hc = pow(10, _c + _d * T);

  for (int k = 0; k < MOORE_NEIGHBORS; k++)
  {
    sz0 = GET(Sz, c, i, j);
    sz = GET(Sz, c, i + Xi[k], j + Xj[k]);
    h[k] = GET(Sh, c, i + Xi[k], j + Xj[k]);
    w[k] = Pc;
    Pr[k] = rr;

    if (k < VON_NEUMANN_NEIGHBORS)
      z[k] = sz;
    else
      z[k] = sz0 - (sz0 - sz) / sqrt(2.0);
  }

  H[0] = z[0];
  theta[0] = 0;
  eliminated[0] = false;
  for (int k = 1; k < MOORE_NEIGHBORS; k++)
    if (z[0] + h[0] > z[k] + h[k])
    {
      H[k] = z[k] + h[k];
      theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]);
      eliminated[k] = false;
    }
    else
    {
      // H[k] = 0;
      // theta[k] = 0;
      eliminated[k] = true;
    }

  do
  {
    loop = false;
    avg = h[0];
    counter = 0;
    for (int k = 0; k < MOORE_NEIGHBORS; k++)
      if (!eliminated[k])
      {
        avg += H[k];
        counter++;
      }
    if (counter != 0)
      avg = avg / double(counter);
    for (int k = 0; k < MOORE_NEIGHBORS; k++)
      if (!eliminated[k] && avg <= H[k])
      {
        eliminated[k] = true;
        loop = true;
      }
  } while (loop);

  for (int k = 1; k < MOORE_NEIGHBORS; k++)
    if (!eliminated[k] && h[0] > hc * cos(theta[k]))
      BUF_SET(Mf, r, c, k - 1, i, j, Pr[k] * (avg - H[k]));
    else
      BUF_SET(Mf, r, c, k - 1, i, j, 0.0);
}

void massBalance(
    int i,
    int j,
    int r,
    int c,
    int *Xi,
    int *Xj,
    double *Sh,
    double *Sh_next,
    double *ST,
    double *ST_next,
    double *Mf)
{
  const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3, 2, 1, 0, 6, 7, 4, 5};
  double inFlow;
  double outFlow;
  double neigh_t;
  double initial_h = GET(Sh, c, i, j);
  double initial_t = GET(ST, c, i, j);
  double h_next = initial_h;
  double t_next = initial_h * initial_t;

  for (int n = 1; n < MOORE_NEIGHBORS; n++)
  {
    neigh_t = GET(ST, c, i + Xi[n], j + Xj[n]);
    inFlow = BUF_GET(Mf, r, c, inflowsIndices[n - 1], i + Xi[n], j + Xj[n]);

    outFlow = BUF_GET(Mf, r, c, n - 1, i, j);

    h_next += inFlow - outFlow;
    t_next += (inFlow * neigh_t - outFlow * initial_t);
  }

  if (h_next > 0)
  {
    t_next /= h_next;
    SET(ST_next, c, i, j, t_next);
    SET(Sh_next, c, i, j, h_next);
  }
}

void computeNewTemperatureAndSolidification(
    int i,
    int j,
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
    double *Sz,
    double *Sz_next,
    double *Sh,
    double *Sh_next,
    double *ST,
    double *ST_next,
    double *Mf,
    double *Mhs,
    bool *Mb)
{
  double nT, aus;
  double z = GET(Sz, c, i, j);
  double h = GET(Sh, c, i, j);
  double T = GET(ST, c, i, j);

  if (h > 0 && GET(Mb, c, i, j) == false)
  {
    aus = 1.0 + (3 * pow(T, 3.0) * Pepsilon * Psigma * Pclock * Pcool) / (Prho * Pcv * h * Pac);
    nT = T / pow(aus, 1.0 / 3.0);

    if (nT > PTsol) // no solidification
      SET(ST_next, c, i, j, nT);
    else // solidification
    {
      SET(Sz_next, c, i, j, z + h);
      SET(Sh_next, c, i, j, 0.0);
      SET(ST_next, c, i, j, PTsol);
      SET(Mhs, c, i, j, GET(Mhs, c, i, j) + h);
    }
  }
}

void boundaryConditions(int i, int j,
                        int r,
                        int c,
                        double *Mf,
                        bool *Mb,
                        double *Sh,
                        double *Sh_next,
                        double *ST,
                        double *ST_next)
{
  if (GET(Mb, c, i, j))
  {
    SET(Sh_next, c, i, j, 0.0);
    SET(ST_next, c, i, j, 0.0);
  }
  return;
}

double reduceAdd(int r, int c, double *buffer)
{
  double sum = 0.0;
  for (int i = 0; i < r; i++)
    for (int j = 0; j < c; j++)
      sum += GET(buffer, c, i, j);

  return sum;
}


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

void deallocateSubstates_proj(Sciara *sciara)
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

void printSciaraConfig(Sciara* sciara) {
    if (sciara == NULL) {
        printf("\n[ERROR] Oggetto Sciara nullo!\n");
        return;
    }

    printf("\n================ SCIARA CONFIGURATION DUMP ================\n");

    // --- DOMAIN ---
    if (sciara->domain) {
        printf("\n[DOMAIN]\n");
        printf("  Rows: %d\n", sciara->domain->rows);
        printf("  Cols: %d\n", sciara->domain->cols);
    } else {
        printf("\n[DOMAIN] is NULL\n");
    }

    // --- PARAMETERS ---
    if (sciara->parameters) {
        printf("\n[PARAMETERS]\n");
        printf("  Pclock (Clock):         %lf\n", sciara->parameters->Pclock);
        printf("  Pc (Cell Side):         %lf\n", sciara->parameters->Pc);
        printf("  Pac (Cell Area):        %lf\n", sciara->parameters->Pac);
        printf("  PTsol (Solid. Temp):    %lf\n", sciara->parameters->PTsol);
        printf("  PTvent (Vent Temp):     %lf\n", sciara->parameters->PTvent);
        printf("  Pr_Tsol:                %lf\n", sciara->parameters->Pr_Tsol);
        printf("  Pr_Tvent:               %lf\n", sciara->parameters->Pr_Tvent);
        printf("  a (Param):              %lf\n", sciara->parameters->a);
        printf("  b (Param):              %lf\n", sciara->parameters->b);
        printf("  Phc_Tsol:               %lf\n", sciara->parameters->Phc_Tsol);
        printf("  Phc_Tvent:              %lf\n", sciara->parameters->Phc_Tvent);
        printf("  c (Param):              %lf\n", sciara->parameters->c);
        printf("  d (Param):              %lf\n", sciara->parameters->d);
        printf("  Pcool:                  %lf\n", sciara->parameters->Pcool);
        printf("  Prho (Density):         %lf\n", sciara->parameters->Prho);
        printf("  Pepsilon (Emissivity):  %lf\n", sciara->parameters->Pepsilon);
        printf("  Psigma (Boltzmann):     %le\n", sciara->parameters->Psigma); // %le per numeri molto piccoli
        printf("  Pcv (Specific Heat):    %lf\n", sciara->parameters->Pcv);
        printf("  Algorithm ID:           %d\n",  sciara->parameters->algorithm);
    } else {
        printf("\n[PARAMETERS] is NULL\n");
    }

    // --- SIMULATION ---
    if (sciara->simulation) {
        printf("\n[SIMULATION]\n");
        printf("  Step:                   %d\n", sciara->simulation->step);
        printf("  Maximum Steps:          %d\n", sciara->simulation->maximum_steps);
        printf("  Elapsed Time:           %lf\n", sciara->simulation->elapsed_time);
        printf("  Emission Time:          %u\n", sciara->simulation->emission_time);
        printf("  Effusion Duration:      %lf\n", sciara->simulation->effusion_duration);
        printf("  Total Emitted Lava:     %lf\n", sciara->simulation->total_emitted_lava);
        printf("  Stopping Threshold:     %lf\n", sciara->simulation->stopping_threshold);
        printf("  Refreshing Step:        %d\n", sciara->simulation->refreshing_step);
        printf("  Thickness Vis. Thresh:  %lf\n", sciara->simulation->thickness_visual_threshold);
        
        // Vents Info
        printf("  Number of Vents:        %lu\n", sciara->simulation->vent.size());
        for(size_t i=0; i < sciara->simulation->vent.size(); i++) {
            // Assumo che TVent abbia i metodi x() e y(). Se da errore, commenta queste righe.
            printf("    Vent #%lu -> X: %d, Y: %d\n", i, sciara->simulation->vent[i].x(), sciara->simulation->vent[i].y());
        }

        // Emission Rates Info
        printf("  Emission Rates count:   %lu\n", sciara->simulation->emission_rate.size());
    } else {
        printf("\n[SIMULATION] is NULL\n");
    }

    // --- SUBSTATES (Controllo allocazione) ---
    // Nota: A questo punto i buffer potrebbero non essere ancora allocati se chiami la stampa prima di allocateSubstates_proj
    if (sciara->substates) {
        printf("\n[SUBSTATES POINTERS] (Check if NULL)\n");
        printf("  Sz  (Altitude): %p\n", (void*)sciara->substates->Sz);
        printf("  Sh  (Lava H):   %p\n", (void*)sciara->substates->Sh);
        printf("  ST  (Temp):     %p\n", (void*)sciara->substates->ST);
        printf("  Mf  (Flows):    %p\n", (void*)sciara->substates->Mf);
        printf("  Mb  (Border):   %p\n", (void*)sciara->substates->Mb);
    } else {
        printf("\n[SUBSTATES] is NULL\n");
    }
    
    printf("===========================================================\n\n");
}

int main(int argc, char **argv)
{
    Sciara *sciara;
    init(sciara);

    int max_steps = atoi(argv[MAX_STEPS_ID]);
    loadConfiguration(argv[INPUT_PATH_ID], sciara);

    allocateSubstates_proj(sciara);

    printSciaraConfig(sciara);

    int *d_Xi, *d_Xj;
    cudaMallocManaged((void**)&d_Xi, MOORE_NEIGHBORS * sizeof(int));
    cudaMallocManaged((void**)&d_Xj, MOORE_NEIGHBORS * sizeof(int));
    cudaMemcpy(d_Xi, sciara->X->Xi, MOORE_NEIGHBORS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Xj, sciara->X->Xj, MOORE_NEIGHBORS * sizeof(int), cudaMemcpyHostToDevice);

    int rows = sciara->domain->rows;
    int cols = sciara->domain->cols;
    
    dim3 block(32, 32);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    printf("Inizializzati i blocchi\n");

    double total_current_lava = -1;
    simulationInitialize(sciara);

    printf("Inizializzata la simulazione\n");

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
                emitLava_global(i, j, sciara);
            }
        }


        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, matrixSize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, matrixSize, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        computeOutflows_Global<<<grid, block>>>(sciara);

        massBalance_Global<<<grid, block>>>(sciara);

        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, matrixSize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, matrixSize, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        computeNewTemperatureAndSolidification_Global<<<grid, block>>>(sciara);

        cudaMemcpy(sciara->substates->Sz, sciara->substates->Sz_next, matrixSize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, matrixSize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, matrixSize, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();


        boundaryConditions_Global<<<grid, block>>>(sciara);

        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, matrixSize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, matrixSize, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        if (sciara->simulation->step % reduceInterval == 0)
        {
          printf("valore lava pre step %d, %lf\n", sciara->simulation->step, total_current_lava);
          cudaDeviceSynchronize();
          total_current_lava = reduceAdd(rows, cols, sciara->substates->Sh);
          printf("Fatta reduceAdd a step %d, %lf\n", sciara->simulation->step,total_current_lava);
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
    deallocateSubstates_proj(sciara); 
    finalize(sciara);

    return 0;
}
