#ifndef __ZTrackSoA__
#define __ZTrackSoA__

#include "stdint.h"

struct ZTrackSoA {
   
    static constexpr uint32_t MAXTRACKS = 1024;
    uint32_t evtID;	
    uint32_t ntrks;
    uint16_t itrk[MAXTRACKS];
    float zt[MAXTRACKS];
    float dz2[MAXTRACKS];
    float tip[MAXTRACKS];
    float pt[MAXTRACKS];
    float ndof[MAXTRACKS];
    float chi2[MAXTRACKS];
};

#endif
