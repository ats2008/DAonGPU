#ifndef RecoPixelVertexing_PixelVertexFinding_gpuDAVertexr_h
#define RecoPixelVertexing_PixelVertexFinding_gpuDAVertexr_h

#include "ZVertexSoA.h"
#include "ZTrackSoA.h"
#include "stdint.h"
#include <cstddef>
namespace gpuDAVertexer{
    static constexpr uint32_t MAXTRACKS = ZVertexSoA::MAXTRACKS;
    static constexpr uint32_t MAXVTX = ZVertexSoA::MAXVTX;

    struct Workspace{

 // Track Parameters 
    uint32_t nTracks;
    uint32_t nVertex  ;
    uint32_t temp_nVertex  ;

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
    
    float rhok               [MAXVTX];
    float rhok_numer[MAXTRACKS*MAXVTX];
    float rho_denom                  ;
   
    float rhok_temp     [MAXVTX];
    float zVtx_temp    [MAXVTX];
    
    int dauterMap        [MAXVTX];

    int   hasThermalized[1];
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
        void allocateGPUworkspace();     
	ZVertexSoA* makeAsync(ZTrackSoA * track,int n=20);
    
    private:
        Workspace *wrkspace;
   
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
   //  Temp variables
    size_t   temp_storage_bytes            ;

   };
}

#endif
