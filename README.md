# DAonGPU
Deterministic Anneling fro vertex finding for CMS event reconstruction. This is a standalone repo and does not use cmssw framework other than to generate the .root files with track details that can be used for validation. Dummy csv files could also be loded for validation/deugging/benchmarking.

## Utils
- makeCSVofTracks.cc   : root macro that coverts the \*.root file to csv file

## Basic Workflow
**Development workflow for kernels**
```
git clone git@github.com:ats2008/DAonGPU.git
cd DAonGPU
git checkout da_v0
cd DA_v0
# compiling the code with nvcc on a cuda supported system
nvcc  sketch_DA_v7.cu  -o dav1.exe

# running the binary
./dav1.exe
```


**Extracting General Tracks as csv files from cms GEN-SIM-RECO files**
This wflow saves the reconstructed track and vertex details to a root tree and writes it out to a .root file. (do not know how to acces thye presice gen-particle details !! TODO : to check it out later)
```bash=
#setup a cmssw environment
cd CMSSW_11_2_0/src/
cmsenv
git clone git@github.com:ats2008/DAonGPU.git
cd DAonGPU
cd getTracksAndVtxs
scram b
cd test
# edit getTrakAndVertex_cfg to proper source files
cmsRun getTrakAndVertex_cfg.py
#will generate a TrkVtxData.root file with data
```
**Generating the track.csv and vertices.csv file for validation**
- From  GEN-SIM-RECO files
  - The generated TrkVtxData.root file could be parsed by [`makeCSVofTracks.cc`](https://github.com/ats2008/DAonGPU/blob/mergeDev/utils/makeCSVfromTTree.cc) ( `DAonGPU/utils/makeCSVfromTTree.cc` ) file to generate  .csv files that could be used for validation
  - This can porbably help in debugging various steps.
- From the DummyDatasetGenerator
  - See the jupyter notebook  [`DAonGPU/DummyTrackAndVertexGen.ipynb`](https://github.com/ats2008/DAonGPU/blob/mergeDev/DummyTrackAndVertexGen.ipynb) to see various distributions available for validation

**DA on GPU validation workflow**

With the a track.csv file ready, we can run some demo code ( A sample TrkVtxData.root file , and corresponding track.csv can be found [here](https://cernbox.cern.ch/index.php/s/yeXjXKOnLbvJqZv))
```bash=
git clone git@github.com:ats2008/DAonGPU.git
cd DAonGPU/DAonGPU
git checkout da_v1
cp <path to tracks.csv>/tracks.csv . 
make 
./main.exe >log
cp <path to vertices.csv>/vertices.csv .
```
Run the jupyter notebook [DAonGPU/ValidationForDA.ipynb](https://github.com/ats2008/DAonGPU/blob/mergeDev/ValidationForDA.ipynb) completely to generate validation plots

**Notes**
- The ./main > **log** is important , since validation is done after reading the log file
- The vertices.csv and tracks.csv has to be in the `DAonGPU/DAonGPU` folder
- ( and ofcourse all these could be modified if u edit the notebook appropriately )

