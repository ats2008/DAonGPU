#ifndef RecoPixelVertexing_PixelVertexFinding_gpuDAVertexr_h
#define RecoPixelVertexing_PixelVertexFinding_gpuDAVertexr_h

#include "ZVertexSoA.h"
#include "ZTrackSoA.h"
#include "stdint.h"

namespace gpuDAVertexer{
    static constexpr uint32_t MAXTRACKS = ZVertexSoA::MAXTRACKS;
    static constexpr uint32_t MAXVTX = ZVertexSoA::MAXVTX;

    struct Workspace{

 // Track Parameters 
    uint32_t nTracks;
    uint32_t nVertex  ;

    uint16_t itrk[MAXTRACKS];
    float zt[MAXTRACKS];
    float dz2[MAXTRACKS];
    float pi[MAXTRACKS];

//  DA workspace variables
    float FEnergyA [MAXTRACKS*MAXVTX];
    
    float pik      [MAXTRACKS*MAXVTX];
    float pik_numer[MAXTRACKS*MAXVTX];
    float pik_denom[MAXTRACKS*MAXVTX];
 
    float zk_delta  	    [MAXVTX];
    float zk_numer[MAXTRACKS*MAXVTX];
    float zk_denom[MAXTRACKS*MAXVTX];
	
    float tc                [MAXVTX];	
    float tc_numer[MAXTRACKS*MAXVTX];
    float tc_denom[MAXTRACKS*MAXVTX];
    
  //  float rho     [MAXTRACKS*MAXVTX];
  //  float rho_denom[MAXTRACKS*MAXVTX];
  //  float rho_denom[MAXTRACKS*MAXVTX];

    bool  hasThermalized;
// Vertex Parameters
    float zVtx[MAXVTX];

// DA variables 

    float beta;
    float betaSplitMax;
    float betaMax;
    float betaFactor;
    float maxDZforMerge;

    float Eik[200]; // TODO :  size needs to be fixed, based on the algo.
    
    // and many more to be added as developemnt prgresses
    
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
