#include "main.h"
#include "ZTrackSoA.h"

#include <cuda.h>
#include <cuda_runtime.h>

ZTrackSoA * loadTracksToGPU(std::string csv_fname,int nevts , int evtStart, int evtEnd )
{
	cout<<"\n HA HA in loadTracksToGPU \n";
	
	if(evtStart==-1) evtStart =0;
	if(evtEnd==-1) evtEnd =evtStart+nevts;
	if(evtEnd-evtStart > nevts) evtEnd=evtStart+nevts;

	ZTrackSoA* trackList = new ZTrackSoA[nevts]; 

	fstream csvfile(csv_fname.c_str(),ios::in);
	if(!csvfile)
	{
		cout<<" Unable To open file : "<<csv_fname<<",  Exiting now \n";
		return nullptr;
	}

	vector<string> row;

	string line,word;

	int idx(-1),evtID,trackCount;
	getline(csvfile,line);
	do{
		row.clear();
		cout<<"line = "<<line<<"\n";	
		if(line[0]=='#'){line=""; continue;}
		
		stringstream s(line);
		while(getline(s, word, ','))
		{
			row.push_back(word);
		}

		if(line[0]=='@')
		{
			idx++;
			if(idx>=nevts) break;
			trackCount=0;
			std::cout<<" row : "<<row[0]<<" , "<<row[1]<<"\n";
			trackList[idx].evtID=uint32_t(stoi(row[0].erase(0,1)));
			trackList[idx].ntrks=uint32_t(stoi(row[1]));
		 	line="";
			continue;
		}

		std::cout<<" row : "<<row[0]<<"+"<<row[1]<<"+"<<row[2]<<"+"<<row[3]<<"+"<<row[4]<<"+"<<row[5]<<"\n";
	  	trackList[idx].itrk[trackCount]=  uint16_t(stoi(row[0]));
	  	trackList[idx].zt[trackCount]  =  float(stof(row[1]));
	  	trackList[idx].dz2[trackCount] =  float(stof(row[2]));
	  	trackList[idx].tip[trackCount] =  float(stof(row[3]));
	  	trackList[idx].pt[trackCount]  =  float(stof(row[4]));
	  	trackList[idx].ndof[trackCount]  =  float(stof(row[5]));
	  	trackList[idx].chi2[trackCount]=  float(31.415);
		std::cout<<" data : "<<trackList[idx].itrk[trackCount]<<"+";
		std::cout<<trackList[idx].zt[trackCount]<<"+";
		std::cout<<trackList[idx].dz2[trackCount]<<"+";
		std::cout<<trackList[idx].tip[trackCount]<<"+";
		std::cout<<trackList[idx].pt[trackCount]<<"+";
		std::cout<<trackList[idx].ndof[trackCount]<<"\n";
		std::cout<<"\n";
		trackCount++;
		line="";
	
	}while(getline(csvfile,line));
	csvfile.close();
	
	ZTrackSoA * tracksOnGPU;
	cout<<sizeof(ZTrackSoA)*nevts<<"\n";
	cudaMalloc(&tracksOnGPU,sizeof(ZTrackSoA)*nevts);
	cudaMemcpy(tracksOnGPU,trackList,sizeof(ZTrackSoA)*nevts,cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	return tracksOnGPU;

}

