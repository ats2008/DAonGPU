
#include "stdint.h"

struct ZTrackSoA {
   
    static constexpr uint32_t MAXTRACKS = 32 * 1024;
    uint32_t ntrks;
    uint16_t itrk[MAXTRACKS];
    float zt[MAXTRACKS];
    float dz2[MAXTRACKS];
    float pi[MAXTRACKS];
};

