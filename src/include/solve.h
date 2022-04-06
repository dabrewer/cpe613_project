#ifndef _SOLVE_H_
#define _SOLVE_H_

#include <stdint.h>

#define ACCEL_FACTOR    1.0
#define PRECISION       0.001

void init(uint16_t size);
void deinit();
void solve();
void save(const char *fname);

#endif //_SOLVE_H_