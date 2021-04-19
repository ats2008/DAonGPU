void makeCSVofTracks(TString fname,int N)
{
	
	static constexpr uint32_t MAXTRACKS=60000;
	static constexpr uint32_t MAXVTX   =50000;
	TFile file(fname,"READ");
	auto trackTree = (TTree*) file.Get("getTrackAndVtxData/Track");
	auto vtxTree = (TTree*) file.Get("getTrackAndVtxData/Vertex");
	
	float zTrack[MAXTRACKS];
	float zErrTrack[MAXTRACKS];
	float ndofTrack[MAXTRACKS];
	float ptTrack[MAXTRACKS];
	float tipTrack[MAXTRACKS];

	float zVtx[MAXVTX];
	float dZ2Vtx[MAXVTX];
	
	int ntracks,nvtxs;

	trackTree->SetBranchAddress("ntracks",&ntracks);
	trackTree->SetBranchAddress("zTrack",zTrack);
	trackTree->SetBranchAddress("zErrTrack",zErrTrack);
	trackTree->SetBranchAddress("ndofTrack",ndofTrack);
	trackTree->SetBranchAddress("ptTrack",ptTrack);
	trackTree->SetBranchAddress("tipTrack",tipTrack);
	
	vtxTree->SetBranchAddress("nvtxs",&nvtxs);
	vtxTree->SetBranchAddress("zVertex",zVtx);
	vtxTree->SetBranchAddress("zErrVertex",dZ2Vtx);

	int Na=trackTree->GetEntries();

	fstream csvfile("tracks.csv",ios::out);
	csvfile<<"#   @evt,numtracks"<<"\n";
	csvfile<<"#   track_idx,ztrack,zErrTrack,tipTrack,ptTrack,ndofTrack"<<"\n";
	
	fstream csvfile_vtx("vertices.csv",ios::out);
	csvfile_vtx<<"#   @evt,numVtxs"<<"\n";
	csvfile_vtx<<"#   vtxIdx,zVtx,dZ2Vtx"<<"\n";


	for(int i=0;i<N;i++)
	{
		if(i%400==0)
			cout<<"Doing "<<i<<"\n";
		trackTree->GetEntry(i);
		vtxTree->GetEntry(i);

		csvfile<<"@"<<i<<","<<ntracks<<"\n";
		csvfile_vtx<<"@"<<i<<","<<nvtxs<<"\n";
		
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
	
         	for(int j=0;j<nvtxs;j++)
		{
			csvfile_vtx<<j<<",";
			csvfile_vtx<<zVtx[j]<<",";
			csvfile_vtx<<dZ2Vtx[j]<<",";
			csvfile_vtx<<"\n";
		}
	
	}
	
	csvfile.close();
	csvfile_vtx.close();
	file.Close();
}
