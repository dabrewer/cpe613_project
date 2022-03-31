#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <mesh.h>

using namespace std;

// ################################################################################
// DEFINITIONS
// ################################################################################
// Define macro for easier 3d memory access
#define POTENTIALS(x,y,z) potentials[(z * _x_size * _y_size) + (y * _x_size) + x] 

// ################################################################################
// MESH
// ################################################################################
Mesh::Mesh( const uint16_t mesh_size )
{
    _potentials_initialized = false;

    _x_size = mesh_size;
    _y_size = mesh_size;
    _z_size = mesh_size;

    init();

    cout << "x_size: " << _x_size << endl;
    cout << "y_size: " << _y_size << endl;
    cout << "z_size: " << _z_size << endl;
}

Mesh::Mesh( const uint16_t x_size, const uint16_t y_size, const uint16_t z_size )
{
    _potentials_initialized = false;

    _x_size = x_size;
    _y_size = y_size;
    _z_size = z_size;

    init();

    cout << "x_size: " << _x_size << endl;
    cout << "y_size: " << _y_size << endl;
    cout << "z_size: " << _z_size << endl;
}

Mesh::~Mesh()
{
    delete potentials;
}

void Mesh::init()
{
    // Only allocate memory once
    if( _potentials_initialized == false)
    {
        _potentials_initialized = true;
        // Dynamically allocate space for array
        potentials = new Voxel[getNumVoxels()];
        // All voxels will be initalized to 0.0V by default
    }

    // NOTE: this call can be modified later to accomodate more geometries
    initBoundaries();
    // Initialize boundary values for parallel plate cap
    initCapacitor();
}

void Mesh::initBoundaries()
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

void Mesh::initCapacitor()
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

uint32_t Mesh::getNumVoxels()
{
    return _y_size * _x_size * _z_size;
}

void Mesh::save( const char *fname )
{
    // FILE *fp = fopen(fname, "w");
    // for(uint16_t y = 0; y < _y_size; y++)
    // {
    //     for(uint16_t x = 0; x < _x_size; x++)
    //     {
    //         fprintf(fp, "%lf\t%lf\t%lf\n", (double)(x * _mesh_size), (double)(y * _mesh_size), POTENTIALS(x,y).getValue());
    //         //printf("%lf %lf %lf %d\n", (double)(x * _mesh_size), (double)(y * _mesh_size), POTENTIALS(x,y).getValue(), POTENTIALS(x,y).isBoundary());
    //     }
    // }
}

double Mesh::sorResidual(uint16_t x, uint16_t y, uint16_t z)
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

// TODO: SOR NEEDS TO RETURN NODE, NOT DOUBLE, SINCE IT IS FILLING A NODE ARRAY
// ALSO, WE NEED TO RETAIN IS BOUNDARY DATA WHEN WE RETURN BOUNDARY NODES TOO
Voxel Mesh::sor(float accel_factor, uint16_t i)
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

    newValue = voxel.getValue() + (accel_factor / 6.0) * sorResidual(x, y, z);

    return Voxel(newValue, voxel.isBoundary());
}

double Mesh::iterate( float accel_factor )
{
    double maxError = 0.0f;

    Voxel *potentials_shadow;
    potentials_shadow = new Voxel[getNumVoxels()];   

    double error;
    for(uint16_t i = 0; i < getNumVoxels(); i++)
    {
        potentials_shadow[i] = sor(accel_factor, i);

        error = fabs(potentials_shadow[i].getValue() - potentials[i].getValue());

        if(error > maxError)
            maxError = error;
    }

    delete potentials;
    potentials = potentials_shadow;

    //cout << maxError << endl;

    return maxError;
}
