#include <iostream>
#include <string.h>
#include <fstream>
#include <sys/time.h>
#include <stdlib.h>

#include <solve.h>

using namespace std;

// ################################################################################
// CMD LINE ARG DEFINITIONS
// ################################################################################
#define NUM_ARGS    5
#define USAGE       "./bin/main [MESH_SIZE] [FNAME_MESH_OUT] [FNAME_FIELD_OUT] [FNAME_STAT_OUT]"
#define MESH_SIZE   argv[1]
#define FNAME_MESH_OUT   argv[2]
#define FNAME_FIELD_OUT   argv[3]
#define FNAME_STAT_OUT   argv[4]

// ################################################################################
// TIMER COMPONENTS
// Adapted from UAH CPE-512 heat_2d_serial.c 
// Written by Dr. Buren Wells
// ################################################################################
#define TIMER_CLEAR     (tv1.tv_sec = tv1.tv_usec = tv2.tv_sec = tv2.tv_usec = 0)
#define TIMER_START     gettimeofday(&tv1, (struct timezone*)0)
#define TIMER_ELAPSED   (double) (tv2.tv_usec- tv1.tv_usec)/1000000.0+(tv2.tv_sec-tv1.tv_sec)
#define TIMER_STOP      gettimeofday(&tv2, (struct timezone*)0)
struct timeval tv1,tv2;

// Private function declarations
void initBoundaries();
void initCapacitor();
__device__ float sor(uint16_t i, uint16_t x, uint16_t y, uint16_t z, float *potentials, float *potentials_shadow, bool *isBoundary, uint16_t _x_size, uint16_t _y_size, uint16_t _z_size);
__device__ float residual(uint16_t x, uint16_t y, uint16_t z, float *potentials, float *potentials_shadow, uint16_t _x_size, uint16_t _y_size, uint16_t _z_size);
__global__ void solveKernel(float *potentials, float *potentials_shadow, bool *isBoundary, uint16_t _x_size, uint16_t _y_size, uint16_t _z_size);

// Define macro for easier 3d memory access
#define GET_INDEX(x,y,z) (((z) * _x_size * _y_size) + ((y) * _x_size) + (x))

// ################################################################################
// MAIN
// ################################################################################
int main( int argc, char *argv[] )
{
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

    //uint16_t iterations;
    ofstream statFile;

    if(argc != NUM_ARGS)
    {
        cout << USAGE << endl;
        return -1;
    }

    if(atoi(MESH_SIZE) == 0 || atoi(MESH_SIZE) < 10)
    {
        cout << "Mesh size must be an integer >= 10";
        return -1;
    }

    // Initialize 3D voltage mesh representing physical geometry
    cout << atoi(MESH_SIZE) << endl;
    cout << "Initializing Mesh..." << endl;
    //init(atoi(MESH_SIZE));

    // ################################################################################
    // INIT CODE START
    // ################################################################################
    cout << "dbg-3";
    _x_size = MESH_SIZE;
    cout << "dbg-2";
    _y_size = MESH_SIZE;
    cout << "dbg-1";
    _z_size = MESH_SIZE;

    cout << "dbg0";
    numVoxels = _x_size * _y_size * _z_size;

    cout << "dbg1";
    cudaMallocManaged(&potentials,numVoxels*sizeof(float));
    cout << "dbg2";
    cudaMallocManaged(&potentials_shadow,numVoxels*sizeof(float));
    cout << "dbg3";
    cudaMallocManaged(&isBoundary,numVoxels*sizeof(bool));
    cout << "dbg4";

    //initBoundaries();
    // ################################################################################
    // INIT BOUNDARIES CODE START
    // ################################################################################

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
    // ################################################################################
    // INIT BOUNDARIES CODE END
    // ################################################################################

    cout << "dbg5";
    //initCapacitor();
    // ################################################################################
    // INIT CAP CODE START
    // ################################################################################
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
    // ################################################################################
    // INIT CAP CODE END
    // ################################################################################
    cout << "dbg6";

    dimGrid = dim3(1,1,1);
    dimBlock = dim3(_x_size, _y_size, _z_size);
    // ################################################################################
    // INIT CODE END
    // ################################################################################

    // Start iteration
    cout << "Starting Iteration..." << endl;
    TIMER_CLEAR;
    TIMER_START;
    
    //iterations = 0;
    //solve();
    // ################################################################################
    // SOLVE CODE END
    // ################################################################################
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
    // ################################################################################
    // SOLVE CODE END
    // ################################################################################
    
    TIMER_STOP;

    // Display run information on screen
    cout << "Saving Results..." << endl;
    //cout << "Iterations: " << iterations << endl;
    cout << "Elapsed Time: " << TIMER_ELAPSED << endl;

    // Save mesh model to output file
    //save(FNAME_MESH_OUT, FNAME_FIELD_OUT);
    // ################################################################################
    // SAVE CODE START
    // ################################################################################
    FILE *fp;

    // Save Potentials
    fp = fopen(FNAME_MESH_OUT, "w");
    for(uint32_t i = 0; i < numVoxels; i++)
    {
        fprintf(fp, "%lf\n", potentials[i]);
    }
    fclose(fp);

    // Save Fields
    fp = fopen(FNAME_FIELD_OUT, "w");
    for(uint32_t i = 0; i < numVoxels; i++)
    {
        float field[3];
       // potentials[i].getField(field);
        fprintf(fp, "%lf\t%lf\t%lf\n", field[0], field[1], field[2]);
    }
    fclose(fp);
    // ################################################################################
    // SAVE CODE END
    // ################################################################################

    // Append run statistics to output file
    statFile.open(FNAME_STAT_OUT, ios_base::app);
    statFile << 'S' << "," << MESH_SIZE << "," << /*iterations << "," <<*/ TIMER_ELAPSED << endl;

    //deinit();
    // ################################################################################
    // DEINIT CODE START
    // ################################################################################
    cudaFree(potentials);
    cudaFree(potentials_shadow);
    cudaFree(isBoundary);
    // ################################################################################
    // DEINIT CODE END
    // ################################################################################

}

__device__ float sor(uint16_t i, uint16_t x, uint16_t y, uint16_t z, float *potentials, float *potentials_shadow, bool *isBoundary, uint16_t _x_size, uint16_t _y_size, uint16_t _z_size)
{
    //Voxel voxel;
    // uint16_t x;
    // uint16_t y;
    // uint16_t z;

    // x = i % _x_size; // TODO:CONVERT i -> x
    // y = (i / _x_size) % _y_size; // TODO:CONVERT i -> y
    // z = (i / _x_size) / _y_size; // TODO:CONVERT i -> z

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