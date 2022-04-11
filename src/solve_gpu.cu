#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <solve.h>

using namespace std;

float maxError;
uint16_t _x_size;
uint16_t _y_size;
uint16_t _z_size;
uint32_t numVoxels;
float *potentials;
float *potentials_shadow;
bool *isBoundary;
dim3 dimGrid;
dim3 dimBlock;

// Private function declarations
void initBoundaries();
void initCapacitor();
__device__ float sor(uint16_t i, uint16_t x, uint16_t y, uint16_t z, float *potentials, float *potentials_shadow, bool *isBoundary, uint16_t _x_size, uint16_t _y_size, uint16_t _z_size);
__device__ float residual(uint16_t x, uint16_t y, uint16_t z, float *potentials, float *potentials_shadow, uint16_t _x_size, uint16_t _y_size, uint16_t _z_size);
__global__ void solveKernel(float *potentials, float *potentials_shadow, bool *isBoundary, uint16_t _x_size, uint16_t _y_size, uint16_t _z_size);

// Define macro for easier 3d memory access
#define GET_INDEX(x,y,z) (((z) * _x_size * _y_size) + ((y) * _x_size) + (x))

#define iceil(num,den) (num+den-1)/den

void init(uint16_t size, uint16_t tile_width_x, uint16_t tile_width_y, uint16_t tile_width_z)
{
    // store x, y, z dimensions
    _x_size = size;
    _y_size = size;
    _z_size = size;
    // calculate total number of voxels
    numVoxels = _x_size * _y_size * _z_size;

    // Allocate unified memory for all arrays
    cudaMallocManaged(&potentials,numVoxels*sizeof(float));
    cudaMallocManaged(&potentials_shadow,numVoxels*sizeof(float));
    cudaMallocManaged(&isBoundary,numVoxels*sizeof(bool));

    // Init environment
    initBoundaries();
    initCapacitor();

    // Init grid dimensions
    dimGrid.x = iceil(_x_size, tile_width_x);
    dimGrid.y = iceil(_y_size, tile_width_y);
    dimGrid.y = iceil(_z_size, tile_width_z);
    // Init block dimensions
    dimBlock = dim3(_x_size, _y_size, _z_size);
}

void deinit()
{
    cudaFree(potentials);
    cudaFree(potentials_shadow);
    cudaFree(isBoundary);
}

void initBoundaries()
{
    // Set x-plane boundaries
    for(int j = 0; j < _y_size; j++)
    {
        for(int k = 0; k < _z_size; k++)
        {
            potentials[GET_INDEX(0,j,k)] = 0.0;
            isBoundary[GET_INDEX(0,j,k)] = true;
            potentials[GET_INDEX((_x_size-1),j,k)] = 0.0;
            isBoundary[GET_INDEX((_x_size-1),j,k)] = true;
        }
    }
    // Set y-plane boundaries
    for(int i = 0; i < _x_size; i++)
    {
        for(int k = 0; k < _z_size; k++)
        {
            potentials[GET_INDEX(i,0,k)] = 0.0;
            isBoundary[GET_INDEX(i,0,k)] = true;
            potentials[GET_INDEX(i,(_y_size-1),k)] = 0.0;
            isBoundary[GET_INDEX(i,(_y_size-1),k)] = true;
        }
    }
    // Set z-plane boundaries
    for(int i = 0; i < _x_size; i++)
    {
        for(int j = 0; j < _y_size; j++)
        {
            potentials[GET_INDEX(i,j,0)] = 0.0;
            isBoundary[GET_INDEX(i,j,0)] = true;
            potentials[GET_INDEX(i,j,(_x_size-1))] = 0.0;
            isBoundary[GET_INDEX(i,j,(_x_size-1))] = true;
        }
    }
}

void initCapacitor()
{
    // Define plate potential
    float plate1_potential = 12.0;
    float plate2_potential = -12.0;
    // Define width common to both plates
    uint16_t x_min = (_x_size / 10) * 3;
    uint16_t x_max = ((_x_size / 10) * 8) - 1;
    // Define depth common to both plates
    uint16_t z_min = (_y_size / 10) * 3;
    uint16_t z_max = ((_y_size / 10) * 8) - 1;
    // Define height of plate 1
    uint16_t y1_min = (_y_size / 10) * 4;
    uint16_t y1_max = ((_y_size / 10) * 5) - 1;
    // Define height of plate 2
    uint16_t y2_min = (_y_size / 10) * 4;
    uint16_t y2_max = ((_y_size / 10) * 5) - 1;

    for(int i = x_min; i <= x_max; i++)
    {
        for(int k = z_min; k <= z_max; k++)
        {
            // Set potentials for plate 1
            for(int j = y1_min; j <= y1_max; j++)
            {
                potentials[GET_INDEX(i,j,k)] = plate1_potential;
                isBoundary[GET_INDEX(i,j,k)] = true;
            }
            // Set potentials for plate 2
            for(int j = y2_min; j <= y2_max; j++)
            {
                potentials[GET_INDEX(i,j,k)] = plate2_potential;
                isBoundary[GET_INDEX(i,j,k)] = true;
            }
        }
    }
}

void solve()
{
    cudaError_t error_id;

    //TODO: make kernel call to find precision and convert to while loop
    for(int i = 0; i < 600; i++)
    {
        solveKernel<<<dimGrid, dimBlock>>>(potentials, potentials_shadow, isBoundary, _x_size, _y_size, _z_size);

        error_id=cudaGetLastError();
        if (error_id != cudaSuccess)
        {
            printf( "Attempted Launch of solveKernel returned %d\n-> %s\n",
            (int)error_id, cudaGetErrorString(error_id) );
            exit(EXIT_FAILURE);
        }

        cudaDeviceSynchronize();

        float *swap = potentials;
        potentials = potentials_shadow;
        potentials_shadow = swap;
    }
}

__global__ void solveKernel(float *potentials, float *potentials_shadow, bool *isBoundary, uint16_t _x_size, uint16_t _y_size, uint16_t _z_size)
{
    uint16_t x = (blockDim.x * blockIdx.x) + threadIdx.x;
    uint16_t y = (blockDim.y * blockIdx.y) + threadIdx.y;
    uint16_t z = (blockDim.z * blockIdx.z) + threadIdx.z;
    uint16_t i = GET_INDEX(x,y,z);

    potentials_shadow[i] = sor(i,x, y, z, potentials, potentials_shadow, isBoundary, _x_size, _y_size, _z_size);

    //error[i] = fabs(potentials_shadow[i] - potentials[i]);
}

__device__ float sor(uint16_t i, uint16_t x, uint16_t y, uint16_t z, float *potentials, float *potentials_shadow, bool *isBoundary, uint16_t _x_size, uint16_t _y_size, uint16_t _z_size)
{
    if(isBoundary[GET_INDEX(x,y,z)])
        return potentials[i];

    return potentials[i] + (ACCEL_FACTOR / 6.0) * residual(x, y, z, potentials, potentials_shadow, _x_size, _y_size, _z_size);
}

__device__ float residual(uint16_t x, uint16_t y, uint16_t z, float *potentials, float *potentials_shadow, uint16_t _x_size, uint16_t _y_size, uint16_t _z_size)
{   
    float rv;

    // Calculate Residual Error in Each Direction
    // Must ensure not to reach outside mesh model
    rv = 0.0;

    // Right Node
    if((x+1) < _x_size)
        rv += potentials[GET_INDEX(x+1,y,z)] - potentials[GET_INDEX(x,y,z)];
    // Left Node
    if((x-1) >= 0)
        rv += potentials[GET_INDEX(x-1,y,z)] - potentials[GET_INDEX(x,y,z)];
    // Top Node
    if((y+1) < _y_size)
        rv += potentials[GET_INDEX(x,y+1,z)] - potentials[GET_INDEX(x,y,z)];
    // Bottom Node
    if((y-1) >= 0)
        rv += potentials[GET_INDEX(x,y-1,z)] - potentials[GET_INDEX(x,y,z)];
    // Front Node
    if((z+1) < _z_size)
        rv += potentials[GET_INDEX(x,y,z+1)] - potentials[GET_INDEX(x,y,z)];
    // Back Node
    if((z-1) >= 0)
        rv += potentials[GET_INDEX(x,y,z-1)] - potentials[GET_INDEX(x,y,z)];

    return rv;
}

void save(const char *pfname, const char *ffname)
{
    FILE *fp;

    // Save Potentials
    fp = fopen(pfname, "w");
    for(uint32_t i = 0; i < numVoxels; i++)
    {
        fprintf(fp, "%lf\n", potentials[i]);
    }
    fclose(fp);

    // Save Fields
    fp = fopen(ffname, "w");
    for(uint32_t i = 0; i < numVoxels; i++)
    {
        float field[3];
       // potentials[i].getField(field);
        fprintf(fp, "%lf\t%lf\t%lf\n", field[0], field[1], field[2]);
    }
    fclose(fp);
}
