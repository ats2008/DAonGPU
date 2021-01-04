#include "fstream"
#include "vector"
#include "string.h"
#include <sstream>
#include "iostream"
#include "ZTrackSoA.h"

using namespace std;

ZTrackSoA* loadTracksToGPU(std::string csv_fname, int nvts=20,int evtStart=0,int evtEnd=20 );

