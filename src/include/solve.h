#ifndef _SOLVE_H_
#define _SOLVE_H_

#include <stdint.h>

#include <voxel.h>

#define ACCEL_FACTOR    1.0

// Define macro for easier 3d memory access
#define POTENTIALS(x,y,z) potentials[((z) * _x_size * _y_size) + ((y) * _x_size) + (x)] 

void init(uint16_t size);
void deinit();
void initBoundaries();
void initCapacitor();
double solve();
Voxel sor(uint16_t i);
double residual(uint16_t x, uint16_t y, uint16_t z);

#endif //_SOLVE_H_