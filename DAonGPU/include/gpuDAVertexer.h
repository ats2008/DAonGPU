#ifndef RecoPixelVertexing_PixelVertexFinding_gpuDAVertexr_h
#define RecoPixelVertexing_PixelVertexFinding_gpuDAVertexr_h

#include "ZVertexSoA.h"
#include "ZTrackSoA.h"
#include "stdint.h"

namespace gpuDAVertexer{

    struct Workspace{
    static constexpr uint32_t MAXTRACKS = ZVertexSoA::MAXTRACKS;
    static constexpr uint32_t MAXVTX = ZVertexSoA::MAXVTX;

 // Track Parameters 
    uint32_t ntrks;
    uint16_t itrk[MAXTRACKS];
    float zt[MAXTRACKS];
    float dz2[MAXTRACKS];
    float pi[MAXTRACKS];

// Vertex Parameters
    float zVtx[MAXVTX];

//    __host__ __device_ void init() { ntrks=0; }

    };

    class DAVertexer{

    public:
        
	DAVertexer( float tmin=0.5); 
        
	ZVertexSoA* makeAsync(ZTrackSoA * track,int n=20);
    
    private:

    // Dummy list of parameters for DA
	float Tmin;
	float Tpurge;
	float Tstop;
	float coolingFactor;
	float d0CutOff;
	float dzCutOff;
	float uniquetrkweight;
	float vertexSize;
	float zmerge;


   };
}

#endif
