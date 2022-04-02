#include <voxel.h>

Voxel::Voxel( void )
{
    _value = 0;
    _isBoundary = false;
}

Voxel::Voxel( double value )
{
    _value = value;
    _isBoundary = false;
}

Voxel::Voxel( double value, bool isBoundary )
{
    _value = value;
    _isBoundary = isBoundary;
}

bool Voxel::isBoundary( void )
{
    return _isBoundary;
}

void Voxel::setValue( double value )
{
    _value = value;
}

double Voxel::getValue( void )
{
    return _value;
}
