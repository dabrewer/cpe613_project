//-------------------------------------------------------------------------
// Included CUDA libraries
//-------------------------------------------------------------------------
#include <stdio.h>

// iceil macro
// returns an integer ceil value where integer numerator is first parameter
// and integer denominator is the second parameter. iceil is the rounded
// up value of numerator/denominator when there is a remainder
// equivalent to ((num%den!=0) ? num/den+1 : num/den)
#define iceil(num,den) (num+den-1)/den 

//#define TILE_WIDTH 16 // block x and y dimensions

__global__ void sorKernel(float *Pd, float *Md, float *Nd, int Mh, int Mw, int Nw) {
    // ===================================================================
    // Calculate Array Indices
    // ===================================================================
    // TODO
//    int Row=blockDim.y*blockIdx.y+threadIdx.y;
//    int Col=blockDim.x*blockIdx.x+threadIdx.x;

    // ===================================================================
    // Create New Voxel from Old
    // ===================================================================
    potentials_shadow[i] = potentials[i];

    // ===================================================================
    // Update Voxel with Residual if Not Boundary
    // ===================================================================
    if(potentials[i].isBoundary() == false)
    {
        potentials[i] += (accel_factor / 6.0) * sorResidual(x, y, z);
    }
}

void iterate_gpu(float *P, float *M, float *N, int Mh, int Mw, int Nw) {
    float *Md, *Nd, *Pd;
    cudaError_t error_id;

    // ===================================================================
    // Allocate and Initialize Unified Main and Shadow Arrays
    // ===================================================================
    // TODO
    //cudaMallocManaged(&x,N*sizeof(float)));
    //cudaMallocManaged(&y,N*sizeof(float)));

    // ===================================================================
    // Init Kernel Execution Config
    // ===================================================================
    // TODO
    // dim3 dimGrid;
    // dimGrid.x = iceil(Nw,TILE_WIDTH); 
    // dimGrid.y = iceil(Mh,TILE_WIDTH); 
    // dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);

    // ===================================================================
    // Launch Kernel
    // ===================================================================
    // TODO
    // MatrixMulKernel<<<dimGrid, dimBlock>>>(Pd, Md, Nd, Mh, Mw, Nw);
    // error_id=cudaGetLastError();
    // if (error_id != cudaSuccess) {
    //     printf( "Attempted Launch of MatriMulKernel returned %d\n-> %s\n",
    //         (int)error_id, cudaGetErrorString(error_id) );
    //     exit(EXIT_FAILURE);
    // }

   // ===================================================================
   // Clean Up Memory
   // ===================================================================
    // error_id=cudaFree(Md);
    // if (error_id != cudaSuccess) {
    //     printf( "Cuda could not free memory Md -- returned %d\n-> %s\n",
    //         (int)error_id, cudaGetErrorString(error_id) );
    //     exit(EXIT_FAILURE);
    //                     }
    // error_id=cudaFree(Nd);
    // if (error_id != cudaSuccess) {
    //     printf( "Cuda could not free memory Nd -- returned %d\n-> %s\n",
    //         (int)error_id, cudaGetErrorString(error_id) );
    //     exit(EXIT_FAILURE);
    // }

}