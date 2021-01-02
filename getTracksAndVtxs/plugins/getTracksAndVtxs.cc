// -*- C++ -*-
//
// Package:    RecoPixelVertexing/getTracksAndVtxs
// Class:      getTracksAndVtxs
//
/**\class getTracksAndVtxs getTracksAndVtxs.cc RecoPixelVertexing/getTracksAndVtxs/plugins/getTracksAndVtxs.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Aravind Thachayath Sugunan
//         Created:  Fri, 01 Jan 2021 13:45:09 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "TTree.h"

#define MAXTRKS_ 60000
#define MAXVTXS_ 60000

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.

using reco::TrackCollection;

class getTracksAndVtxs : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit getTracksAndVtxs(const edm::ParameterSet&);
  ~getTracksAndVtxs();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;  //used to select what tracks to read from configuration file
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;  //used to select what tracks to read from configuration file


  std::map<std::string,TTree*> histContainer_;


  float zTrack[MAXTRKS_];
  float zErrTrack[MAXTRKS_];
  float ndofTrack[MAXTRKS_];
  float ptTrack[MAXTRKS_];
  float tipTrack[MAXTRKS_];
  int ntracks;

  float zVtx[MAXVTXS_];
  float zErrVtx[MAXVTXS_];
  int nvtxs;  
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  edm::ESGetToken<SetupData, SetupRecord> setupToken_;
#endif
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
getTracksAndVtxs::getTracksAndVtxs(const edm::ParameterSet& iConfig)
    : tracksToken_(consumes<reco::TrackCollection>(iConfig.getUntrackedParameter<edm::InputTag>("trksrc"))) ,
    vertexToken_(consumes<reco::VertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("vtxsrc"))),
    histContainer_()
{}

getTracksAndVtxs::~getTracksAndVtxs() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//

// ------------ method called for each event  ------------
void getTracksAndVtxs::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
 
 auto trackList = iEvent.get(tracksToken_);
 ntracks=trackList.size();
 //std::cout<<"ntrks = "<<ntracks<<"  ;\n";
 
 auto vtxList = iEvent.get(vertexToken_);
 nvtxs = vtxList.size();
 //std::cout<<"nvtxs = "<<nvtxs<<"  ;\n";
 int i=0;
 
 for (const auto& track :  trackList) {
      zTrack[i]= track.dz() ;
      zErrTrack[i]=track.dzError();
      ndofTrack[i]=track.ndof();
      ptTrack[i]=track.pt();
      tipTrack[i]=track.dxy();
      i++;
   //   std::cout<<" At track i = "<<i<<"\n";
  }
  
  i=0;
  for( auto & vtx : vtxList ){
    zVtx[i]    = vtx.z();  
    zErrVtx[i] = vtx.zError();
  i++;
  }

  histContainer_["trk"]->Fill();
  histContainer_["vtx"]->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void getTracksAndVtxs::beginJob() {
  // please remove this method if not needed
    
 edm::Service<TFileService> fs;
 histContainer_["trk"]= fs->make<TTree>("Track","Track Parameters");
 histContainer_["vtx"]= fs->make<TTree>("Vertex","Vertex Parameters");
  
 histContainer_["trk"]->Branch("ntracks",&ntracks,"ntracks/I");
 histContainer_["trk"]->Branch("zTrack",&zTrack,"zTrack[ntracks]/F");
 histContainer_["trk"]->Branch("zErrTrack",&zErrTrack,"zErrTrack[ntracks]/F");
 histContainer_["trk"]->Branch("ndofTrack",&ndofTrack,"ndofTrack[ntracks]/F");
 histContainer_["trk"]->Branch("ptTrack",&ptTrack,"ptTrack[ntracks]/F");
 histContainer_["trk"]->Branch("tipTrack",&tipTrack,"tipTrack[ntracks]/F");
  
 histContainer_["vtx"]->Branch("nvtxs",&nvtxs,"nvtxs/I");
 histContainer_["vtx"]->Branch("zVertex",&zVtx,"zVertex[nvtxs]/F");
 histContainer_["vtx"]->Branch("zErrVertex",&zErrVtx,"zErrVertex[nvtxs]/F");

}
// ------------ method called once each job just after ending the event loop  ------------
void getTracksAndVtxs::endJob() {
  // please remove this method if not needed
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void getTracksAndVtxs::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  
  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  desc.add<edm::InputTag>("trksrc",edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("vtxsrc",edm::InputTag("offlinePrimaryVertices"));
  //descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(getTracksAndVtxs);
