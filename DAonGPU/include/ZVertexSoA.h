#include "stdint.h"

struct ZVertexSoA {
  static constexpr uint32_t MAXTRACKS = 1024;
  static constexpr uint32_t MAXVTX = 128;

  int16_t idv[MAXTRACKS];    // vertex index for each associated (original) track  (-1 == not associate)
  float zv[MAXVTX];          // output z-posistion of found vertices
  float wv[MAXVTX];          // output weight (1/error^2) on the above
  float chi2[MAXVTX];        // vertices chi2
  float ptv2[MAXVTX];        // vertices pt^2
  int32_t ndof[MAXTRACKS];   // vertices number of dof (reused as workspace for the number of nearest neighbours FIXME)
  uint16_t sortInd[MAXVTX];  // sorted index (by pt2)  ascending
  uint32_t nvFinal;          // the number of vertices

  //__host__ __device__ void init() { nvFinal = 0; }
};
