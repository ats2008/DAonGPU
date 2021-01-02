import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("PoolSource",
          fileNames = cms.untracked.vstring(
                    '/store/relval/CMSSW_11_2_0/RelValZMM_14/GEN-SIM-RECO/112X_mcRun3_2021_realistic_v13-v1/10000/3bd9f6dc-9eb2-4eec-af7c-a96537e31187.root',
                    '/store/relval/CMSSW_11_2_0/RelValZMM_14/GEN-SIM-RECO/112X_mcRun3_2021_realistic_v13-v1/10000/b9c3be09-986e-4a80-ae36-037072991a10.root',
                    '/store/relval/CMSSW_11_2_0/RelValZMM_14/GEN-SIM-RECO/112X_mcRun3_2021_realistic_v13-v1/10000/cfdd6387-521e-42e6-bdb0-20b731aec636.root',
                    '/store/relval/CMSSW_11_2_0/RelValZMM_14/GEN-SIM-RECO/112X_mcRun3_2021_realistic_v13-v1/10000/d2ba3e2c-8f7e-4faa-9f0f-b3e8d4ce8a1b.root',
                      ),
        dropDescendantsOfDroppedBranches=cms.untracked.bool(False),
        inputCommands = cms.untracked.vstring(
#                 "keep  *_*_*_RECO"
                "keep  *_offlinePrimaryVertices_*_*",
                "keep  *_generalTracks_*_*",
                )

          )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.maxEvents = cms.untracked.PSet( 
            input = cms.untracked.int32(-1),
                        )
    
process.getTrackAndVtxData = cms.EDAnalyzer("getTracksAndVtxs",
           trksrc = cms.untracked.InputTag("generalTracks"),                  
           vtxsrc = cms.untracked.InputTag("offlinePrimaryVertices"),
        )

process.TFileService = cms.Service("TFileService",fileName = cms.string('TrkVtxData.root')
                                                                              )
process.p = cms.Path(process.getTrackAndVtxData)
