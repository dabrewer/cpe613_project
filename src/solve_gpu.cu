#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <voxel.h>
#include <solve.h>

using namespace std;

__global__ double maxError;
__global__ uint16_t _x_size;
__global__ uint16_t _y_size;
__global__ uint16_t _z_size;
__global__ uint32_t numVoxels;
__global__ Voxel *potentials;
__global__ Voxel *potentials_shadow;

// Private function declarations
void initBoundaries();
void initCapacitor();
__device__ Voxel sor(uint16_t i);
__device__ double residual(uint16_t x, uint16_t y, uint16_t z);
__global__ void solveKernel(Voxel *potentials, Voxel *potentials_shadow);


void init(uint16_t size)
{
    maxError = 0.0;

    _x_size = size;
    _y_size = size;
    _z_size = size;

    numVoxels = _x_size * _y_size * _z_size;

    cudaMallocManaged(&potentials,numVoxels*sizeof(Voxel)));
    cudaMallocManaged(&potentials_shadow,numVoxels*sizeof(Voxel)));

    initBoundaries();
    initCapacitor();

    // dim3 dimGrid;
    // dimGrid.x = iceil(Nw,TILE_WIDTH); 
    // dimGrid.y = iceil(Mh,TILE_WIDTH); 
    // dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);
}

void deinit()
{
    cudaFree(potentials);
    cudaFree(potentials_shadow);
}

void initBoundaries()
{
    // Set x-plane boundaries
    for(int j = 0; j < _y_size; j++)
    {
        for(int k = 0; k < _z_size; k++)
        {
            POTENTIALS(0,j,k) = Voxel(0.0, true);
            POTENTIALS((_x_size-1),j,k) = Voxel(0.0, true);
        }
    }
    // Set y-plane boundaries
    for(int i = 0; i < _x_size; i++)
    {
        for(int k = 0; k < _z_size; k++)
        {
            POTENTIALS(i,0,k) = Voxel(0.0, true);
            POTENTIALS(i,(_y_size-1),k) = Voxel(0.0, true);
        }
    }
    // Set z-plane boundaries
    for(int i = 0; i < _x_size; i++)
    {
        for(int j = 0; j < _y_size; j++)
        {
            POTENTIALS(i,j,0) = Voxel(0.0, true);
            POTENTIALS(i,j,(_x_size-1)) = Voxel(0.0, true);
        }
    }
}

void initCapacitor()
{
    // Define plate potential
    double plate1_potential = 12.0;
    double plate2_potential = -12.0;
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
                POTENTIALS(i,j,k) = Voxel(plate1_potential, true);
            }
            // Set potentials for plate 2
            for(int j = y2_min; j <= y2_max; j++)
            {
                POTENTIALS(i,j,k) = Voxel(plate2_potential, true);
            }
        }
    }
}

void solve()
{
    cudaError_t error_id;

    solveKernel<<<dimGrid, dimBlock>>>(Pd, Md, Nd, Mh, Mw, Nw);

    error_id=cudaGetLastError();
    if (error_id != cudaSuccess)
    {
        printf( "Attempted Launch of solveKernel returned %d\n-> %s\n",
        (int)error_id, cudaGetErrorString(error_id) );
        exit(EXIT_FAILURE);
    }
}

__global__ void solveKernel(Voxel *potentials, Voxel *potentials_shadow, uint16_t _x_size, uint16_t _y_size, uint16_t _z_size)
{
    double maxError;
    double error;

    do
    {
        maxError = 0;
        for(uint16_t i = 0; i < numVoxels; i++)
        {
            potentials_shadow[i] = sor(i);

            error = fabs(potentials_shadow[i].getValue() - potentials[i].getValue());

            if(error > maxError)
                maxError = error;
        }

        Voxel *swap = potentials;
        potentials = potentials_shadow;
        potentials_shadow = swap;
    } while(maxError > PRECISION);
}

__device__ Voxel sor(uint16_t i)
{
    Voxel voxel;
    uint16_t x;
    uint16_t y;
    uint16_t z;
    double newValue;

    x = i % _x_size; // TODO:CONVERT i -> x
    y = (i / _x_size) % _y_size; // TODO:CONVERT i -> y
    z = (i / _x_size) / _y_size; // TODO:CONVERT i -> z

    voxel = POTENTIALS(x,y,z);

    if(voxel.isBoundary())
        return voxel;

    newValue = voxel.getValue() + (ACCEL_FACTOR / 6.0) * residual(x, y, z);

    return Voxel(newValue, voxel.isBoundary());
}

__device__ double residual(uint16_t x, uint16_t y, uint16_t z)
{   
    double rv;

    // Calculate Residual Error in Each Direction
    // Must ensure not to reach outside mesh model
    rv = 0.0;

    // Right Node
    if((x+1) < _x_size)
        rv += POTENTIALS(x+1,y,z).getValue() - POTENTIALS(x,y,z).getValue();
    // Left Node
    if((x-1) >= 0)
        rv += POTENTIALS(x-1,y,z).getValue() - POTENTIALS(x,y,z).getValue();
    // Top Node
    if((y+1) < _y_size)
        rv += POTENTIALS(x,y+1,z).getValue() - POTENTIALS(x,y,z).getValue();
    // Bottom Node
    if((y-1) >= 0)
        rv += POTENTIALS(x,y-1,z).getValue() - POTENTIALS(x,y,z).getValue();
    // Front Node
    if((z+1) < _z_size)
        rv += POTENTIALS(x,y,z+1).getValue() - POTENTIALS(x,y,z).getValue();
    // Back Node
    if((z-1) >= 0)
        rv += POTENTIALS(x,y,z-1).getValue() - POTENTIALS(x,y,z).getValue();

    return rv;
}