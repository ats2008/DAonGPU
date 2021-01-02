void makeCSVofTracks(TString fname)
{
	
	static constexpr uint32_t MAXTRACKS=60000;
	TFile file(fname,"READ");
	auto trackTree = (TTree*) file.Get("getTrackAndVtxData/Track");
	
	float zTrack[MAXTRACKS];
	float zErrTrack[MAXTRACKS];
	float ndofTrack[MAXTRACKS];
	float ptTrack[MAXTRACKS];
	float tipTrack[MAXTRACKS];

	int ntracks;

	trackTree->SetBranchAddress("ntracks",&ntracks);
	trackTree->SetBranchAddress("zTrack",zTrack);
	trackTree->SetBranchAddress("zErrTrack",zErrTrack);
	trackTree->SetBranchAddress("ndofTrack",ndofTrack);
	trackTree->SetBranchAddress("ptTrack",ptTrack);
	trackTree->SetBranchAddress("tipTrack",tipTrack);

	int N=trackTree->GetEntries();

	fstream csvfile("tracks.csv",ios::out);
	csvfile<<"#   @evt,numtracks"<<"\n";
	csvfile<<"#   track_idx,ztrack,zErrTrack,tipTrack,ptTrack,ndofTrack"<<"\n";

	for(int i=0;i<N;i++)
	{
		trackTree->GetEntry(i);
		csvfile<<"@"<<i<<","<<ntracks<<"\n";
		for(int j=0;j<ntracks;j++)
		{
			csvfile<<j<<",";
			csvfile<<zTrack[j]<<",";
			csvfile<<zErrTrack[j]<<",";
			csvfile<<tipTrack[j]<<",";
			csvfile<<ptTrack[j]<<",";
			csvfile<<ndofTrack[j];
			csvfile<<"\n";
		}
	
	}
	
	csvfile.close();
	file.Close();
}
