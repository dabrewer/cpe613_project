#ifndef _VOXEL_H_
#define _VOXEL_H_

#include <stdbool.h>

class Voxel
{
public:
    Voxel( void );
    Voxel( float value );
    Voxel( float value, bool isBoundary );
    bool isBoundary( void );
    void setValue( float value );
    float getValue( void );

private:
    float _value;
    bool _isBoundary;
};

#endif //_VOXEL_H_
