#include "iostream"
#include "gpuDAVertexer.h"
#include "main.h"

using namespace std;

int main()
{

	ZTrackSoA * tracksInGPU;

	tracksInGPU=loadTracksToGPU("tracks.csv");
	
	gpuDAVertexer::DAVertexer demoVertexer(5.5);
	
	demoVertexer.makeAsync(tracksInGPU,20);

	return 0;
}



