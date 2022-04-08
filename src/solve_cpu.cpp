#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <voxel.h>
#include <solve.h>

using namespace std;

uint16_t _x_size;
uint16_t _y_size;
uint16_t _z_size;
uint32_t numVoxels;
Voxel *potentials;
Voxel *potentials_shadow;

// Private function declarations
void initBoundaries();
void init_capacitor();
Voxel sor(uint16_t i);
float residual(uint16_t x, uint16_t y, uint16_t z);
void solve_potential();
void solve_field();
void calc_field(uint16_t x, uint16_t y, uint16_t z, float *dV);

// Define macro for easier 3d memory access
#define POTENTIALS(x,y,z) potentials[((z) * _x_size * _y_size) + ((y) * _x_size) + (x)] 

void init(uint16_t size)
{
    _x_size = size;
    _y_size = size;
    _z_size = size;

    numVoxels = _x_size * _y_size * _z_size;

    potentials = new Voxel[numVoxels];
    potentials_shadow = new Voxel[numVoxels];

    initBoundaries();
    init_capacitor();
}

void deinit()
{
    delete potentials;
    delete potentials_shadow;
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

void init_capacitor()
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
    uint16_t y1_min = (_y_size / 10) * 3;
    uint16_t y1_max = ((_y_size / 10) * 4) - 1;
    // Define height of plate 2
    uint16_t y2_min = (_y_size / 10) * 6;
    uint16_t y2_max = ((_y_size / 10) * 7) - 1;

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
    solve_potential();
    solve_field();
}

void solve_potential()
{
    uint16_t iterations;
    float maxError;
    float error;

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

void solve_field()
{
    for(uint16_t i = 0; i < numVoxels; i++)
    {
        float dV[3];
        uint16_t x;
        uint16_t y;
        uint16_t z;

        x = i % _x_size; // TODO:CONVERT i -> x
        y = (i / _x_size) % _y_size; // TODO:CONVERT i -> y
        z = (i / _x_size) / _y_size; // TODO:CONVERT i -> z

        calc_field(x, y, z, dV);

        POTENTIALS(x,y,z).setField(dV);
    }
}

void calc_field(uint16_t x, uint16_t y, uint16_t z, float *dV)
{
    // Approximate Field with Stencil Computation
    // Must ensure not to reach outside mesh model
    dV[0] = 0.0; // x
    dV[1] = 0.0; // y
    dV[2] = 0.0; // z

    // Calculate dV_dx
    // Consider both boundary conditions as special cases
    if((x > 0) && (x < (_x_size-1)))
        dV[0] = POTENTIALS(x-1,y,z).getValue() - POTENTIALS(x+1,y,z).getValue();
    if(x == 0)
        dV[0] = POTENTIALS(x,y,z).getValue() - POTENTIALS(x+1,y,z).getValue();
    if(x == (_x_size-1))
        dV[0] = POTENTIALS(x-1,y,z).getValue() - POTENTIALS(x,y,z).getValue();
    // Calculate dV_dy
    // Consider both boundary conditions as special cases
    if((y > 0) && (y < (_y_size-1)))
        dV[1] = POTENTIALS(x,y-1,z).getValue() - POTENTIALS(x,y+1,z).getValue();
    if(y == 0)
        dV[1] = POTENTIALS(x,y,z).getValue() - POTENTIALS(x,y+1,z).getValue();
    if(y == (_y_size-1))
        dV[1] = POTENTIALS(x,y-1,z).getValue() - POTENTIALS(x,y,z).getValue();
    // Calculate dV_dz
    // Consider both boundary conditions as special cases
    if((z > 0) && (z < (_z_size-1)))
        dV[2] = POTENTIALS(x,y,z-1).getValue() - POTENTIALS(x,y,z+1).getValue();
    if(z == 0)
        dV[2] = POTENTIALS(x,y,z).getValue() - POTENTIALS(x,y,z+1).getValue();
    if(z == (_z_size-1))
        dV[2] = POTENTIALS(x,y,z-1).getValue() - POTENTIALS(x,y,z).getValue();
}

Voxel sor(uint16_t i)
{
    Voxel voxel;
    uint16_t x;
    uint16_t y;
    uint16_t z;
    float newValue;

    x = i % _x_size; // TODO:CONVERT i -> x
    y = (i / _x_size) % _y_size; // TODO:CONVERT i -> y
    z = (i / _x_size) / _y_size; // TODO:CONVERT i -> z

    voxel = POTENTIALS(x,y,z);

    if(voxel.isBoundary())
        return voxel;

    newValue = voxel.getValue() + (ACCEL_FACTOR / 6.0) * residual(x, y, z);

    return Voxel(newValue, voxel.isBoundary());
}

float residual(uint16_t x, uint16_t y, uint16_t z)
{   
    float rv;

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

void save(const char *pfname, const char *ffname)
{
    FILE *fp;

    // Save Potentials
    fp = fopen(pfname, "w");
    for(uint32_t i = 0; i < numVoxels; i++)
    {
        fprintf(fp, "%lf\n", potentials[i].getValue());
    }
    fclose(fp);

    // Save Fields
    fp = fopen(ffname, "w");
    for(uint32_t i = 0; i < numVoxels; i++)
    {
        float field[3];
        potentials[i].getField(field);
        fprintf(fp, "%lf\t%lf\t%lf\n", field[0], field[1], field[2]);
    }
    fclose(fp);
}