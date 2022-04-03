#include <iostream>
#include <string.h>
#include <fstream>
#include <sys/time.h>
#include <stdlib.h>

#include <solve.h>

using namespace std;

// ################################################################################
// CMD LINE ARG DEFINITIONS
// ################################################################################
#define NUM_ARGS    4
#define USAGE       "./bin/main [MESH_SIZE] [FNAME_MESH_OUT] [FNAME_STAT_OUT]"
#define MESH_SIZE   argv[1]
#define FNAME_MESH_OUT   argv[2]
#define FNAME_STAT_OUT   argv[3]

// ################################################################################
// TIMER COMPONENTS
// Adapted from UAH CPE-512 heat_2d_serial.c 
// Written by Dr. Buren Wells
// ################################################################################
#define TIMER_CLEAR     (tv1.tv_sec = tv1.tv_usec = tv2.tv_sec = tv2.tv_usec = 0)
#define TIMER_START     gettimeofday(&tv1, (struct timezone*)0)
#define TIMER_ELAPSED   (double) (tv2.tv_usec- tv1.tv_usec)/1000000.0+(tv2.tv_sec-tv1.tv_sec)
#define TIMER_STOP      gettimeofday(&tv2, (struct timezone*)0)
struct timeval tv1,tv2;

// ################################################################################
// MAIN
// ################################################################################
int main( int argc, char *argv[] )
{
    //uint16_t iterations;
    ofstream statFile;

    if(argc != NUM_ARGS)
    {
        cout << USAGE << endl;
        return -1;
    }

    if(atoi(MESH_SIZE) == 0 || atoi(MESH_SIZE) < 10)
    {
        cout << "Mesh size must be an integer >= 10";
        return -1;
    }

    // Initialize 3D voltage mesh representing physical geometry
    cout << "Initializing Mesh..." << endl;
    init(atoi(MESH_SIZE));

    // Start iteration
    cout << "Starting Iteration..." << endl;
    TIMER_CLEAR;
    TIMER_START;
    
    //iterations = 0;
    solve();
    
    TIMER_STOP;

    // Display run information on screen
    cout << "Saving Results..." << endl;
    //cout << "Iterations: " << iterations << endl;
    cout << "Elapsed Time: " << TIMER_ELAPSED << endl;

    // Save mesh model to output file
    //mesh->save(FNAME_MESH_OUT);

    // Append run statistics to output file
    statFile.open(FNAME_STAT_OUT, ios_base::app);
    statFile << 'S' << "," << MESH_SIZE << "," << /*iterations << "," <<*/ TIMER_ELAPSED << endl;

    deinit();
}
