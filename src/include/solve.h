#ifndef _SOLVE_H_
#define _SOLVE_H_

#include <stdint.h>

#define ACCEL_FACTOR    1.0
#define PRECISION       0.001

#ifdef GPU
void init(uint16_t size, uint16_t tile_width_x, uint16_t tile_width_y, uint16_t tile_width_z);
#else
void init(uint16_t size);
#endif
void deinit();
void solve();
void save(const char *pfname, const char *ffname);

#endif //_SOLVE_H_