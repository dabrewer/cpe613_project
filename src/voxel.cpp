#include <voxel.h>

Voxel::Voxel( void )
{
    _value = 0;
    _isBoundary = false;
}

Voxel::Voxel( float value )
{
    _value = value;
    _isBoundary = false;
}

Voxel::Voxel( float value, bool isBoundary )
{
    _value = value;
    _isBoundary = isBoundary;
}

bool Voxel::isBoundary( void )
{
    return _isBoundary;
}

void Voxel::setValue( float value )
{
    _value = value;
}

float Voxel::getValue( void )
{
    return _value;
}

void Voxel::setField( float *field )
{
    _field[0] = field[0];
    _field[1] = field[1];
    _field[2] = field[2];
}

void Voxel::getField( float *field )
{
    field[0] = _field[0];
    field[1] = _field[1];
    field[2] = _field[2];
}