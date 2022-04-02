#ifndef _VOXEL_H_
#define _VOXEL_H_

#include <stdbool.h>

class Voxel
{
public:
    Voxel( void );
    Voxel( double value );
    Voxel( double value, bool isBoundary );
    bool isBoundary( void );
    void setValue( double value );
    double getValue( void );

private:
    double _value;
    bool _isBoundary;
};

#endif //_VOXEL_H_
