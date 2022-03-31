#ifndef _MESH_H_
#define _MESH_H_

#include <stdint.h>

#include <voxel.h>

#define MAX_VOXELS  100*100*100

class Mesh
{
public:
    Mesh( const uint16_t mesh_size );
    Mesh( const uint16_t x_size, const uint16_t y_size, const uint16_t z_size );
    ~Mesh();
    void init();
    void initBoundaries();
    void initCapacitor();
    uint32_t getNumVoxels( void );
    double iterate( const float accel_factor );
    void save( const char *fname );
private:
    uint16_t _x_size;
    uint16_t _y_size;
    uint16_t _z_size;
    Voxel *ptr_potentials;
    Voxel *ptr_potentials_shadow;
    Voxel potentials0[MAX_VOXELS];
    Voxel potentials1[MAX_VOXELS];
    bool _potentials_initialized;

    Voxel sor( float accel_factor, uint16_t i );
    double sorResidual( uint16_t x, uint16_t y, uint16_t z );
};

#endif // _MESH_H_
