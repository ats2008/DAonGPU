#include "iostream"
#include "gpuDAVertexer.h"
#include "main.h"

using namespace std;

int main()
{

    ZTrackSoA * tracksInGPU;
    const int NEVENTS=2000;

    // ZTrackSoA* loadTracksToGPU(std::string csv_fname, int nvts=20,int evtStart=0,int evtEnd=20 );
    tracksInGPU=loadTracksToGPU("tracks.csv",NEVENTS);

    if(tracksInGPU == nullptr)
        return 1;

    gpuDAVertexer::DAVertexer demoVertexer(5.5); 
    demoVertexer.allocateGPUworkspace();

    for(int i=0; i<NEVENTS; i++)
    {
        cout<<"\n@ Doing vertexing for event , "<<i<<"\n";
        demoVertexer.makeAsync(&tracksInGPU[i],20) ; // the 20 here does not have any meaning , was a prameter passed for testing
    }
    return 0;
}



