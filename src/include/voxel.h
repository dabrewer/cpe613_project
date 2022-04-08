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
    void setField( float *field );
    void getField( float *field );

private:
    float _value;
    float _field[3];
    bool _isBoundary;
};

#endif //_VOXEL_H_
