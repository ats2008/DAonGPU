import numpy as np
import matplotlib.pyplot as plt

# ALL UNITS IN MLLIMETER

def getVtxWithDeltaZs(meanZ,nv_list,deltaZ_mean,deltaZ_width,deltaZ_type):
    nv=np.sum(nv_list)
    if(deltaZ_type=='uniform'):
        vtxZ=np.random.uniform(deltaZ_mean - 0.5*deltaZ_width,deltaZ_mean + 0.5*deltaZ_width,nv)
    if(deltaZ_type=='gauss'):
        vtxZ=np.random.normal(deltaZ_mean,deltaZ_width,nv)
    
    vidx=0
    for vtxCount in nv_list:
        zOffset=0.0
        zSum=0.0
        for i in range(vtxCount):
            t=vtxZ[vidx+i]
            vtxZ[vidx+i]+=zOffset
            zOffset+=t
            zSum+=vtxZ[vidx+i]
        
        vtxZ[vidx:vidx+vtxCount]=vtxZ[vidx:vidx+vtxCount] - zSum/vtxCount +meanZ
#         print("for vtxCount = ",vtxCount," zOffset = ",zOffset)
        vidx+=vtxCount
    return vtxZ

def generateTrkVtxData(NEVENTS=10,genDict={}):
    
    if 'RSEED' in genDict:
        RSEED     =      genDict['RSEED']
    else:
        RSEED     =      0
                      
    if 'PU_DISTRIBUTION_SIGMA' in genDict:
        PU_DISTRIBUTION_SIGMA=  genDict['PU_DISTRIBUTION_SIGMA']
    else:
        PU_DISTRIBUTION_SIGMA  =   5
        
    if 'PU_DISTRIBUTION_MEAN' in genDict:
        PU_DISTRIBUTION_MEAN =  genDict['PU_DISTRIBUTION_MEAN']
    else:
        PU_DISTRIBUTION_MEAN =  20

    if 'VERTEX_Z_DISTRIBUTION_SIGMA' in genDict:
        VERTEX_Z_DISTRIBUTION_SIGMA  =  genDict['VERTEX_Z_DISTRIBUTION_SIGMA']
    else:
        VERTEX_Z_DISTRIBUTION_SIGMA  =  50
    
    if 'VERTEX_Z_DISTRIBUTION_WIDTH' in genDict:
        VERTEX_Z_DISTRIBUTION_WIDTH  =  genDict['VERTEX_Z_DISTRIBUTION_WIDTH']
    else:
        VERTEX_Z_DISTRIBUTION_WIDTH  =  20
                
    if 'VERTEX_Z_DISTRIBUTION_MEAN' in genDict:
        VERTEX_Z_DISTRIBUTION_MEAN   =  genDict['VERTEX_Z_DISTRIBUTION_MEAN']
    else:
        VERTEX_Z_DISTRIBUTION_MEAN   =  0

    if 'TRACK_Z_DISTRIBUTION_SIGMA' in genDict:
        TRACK_Z_DISTRIBUTION_SIGMA   =  genDict['TRACK_Z_DISTRIBUTION_SIGMA']
    else:
        TRACK_Z_DISTRIBUTION_SIGMA   =  0.15
                
    if 'TRACK_Z_DISTRIBUTION_BIAS' in genDict:
        TRACK_Z_DISTRIBUTION_BIAS    =  genDict['TRACK_Z_DISTRIBUTION_BIAS']
    else:
        TRACK_Z_DISTRIBUTION_BIAS    =  0.0 

    if 'TRACKS_PER_VTX_SIGMA' in genDict:
        TRACKS_PER_VTX_SIGMA         =  genDict['TRACKS_PER_VTX_SIGMA']
    else:
        TRACKS_PER_VTX_SIGMA         =  10
                
    if 'TRACKS_PER_VTX_MEAN' in genDict:
        TRACKS_PER_VTX_MEAN          =  genDict['TRACKS_PER_VTX_MEAN']
    else:
        TRACKS_PER_VTX_MEAN          =  60
    
    if 'VERTEX_Z_DISTRIBUTION_TYPE' in genDict:
        VERTEX_Z_DISTRIBUTION_TYPE   = genDict['VERTEX_Z_DISTRIBUTION_TYPE']
    else:
        VERTEX_Z_DISTRIBUTION_TYPE   = 'uniform'
        
    if 'VERTEX_GEN_MODE' in genDict:
        VERTEX_GEN_MODE          = genDict['VERTEX_GEN_MODE']
    else:
        VERTEX_GEN_MODE          = 'Z'
        
    if 'VERTEX_dZ_DISTRIBUTION_TYPE' in genDict:
        VERTEX_dZ_DISTRIBUTION_TYPE   = genDict['VERTEX_dZ_DISTRIBUTION_TYPE']
    else:
        VERTEX_dZ_DISTRUBUTION_TYPE   ='uniform'
    if  'VERTEX_dZ_DISTRIBUTION_MEAN' in genDict:
        VERTEX_dZ_DISTRIBUTION_MEAN   = genDict['VERTEX_dZ_DISTRIBUTION_MEAN']
    else:
        VERTEX_dZ_DISTRIBUTION_MEAN   = 4.0
        
    if  'VERTEX_dZ_DISTRIBUTION_WIDTH' in genDict:
        VERTEX_dZ_DISTRIBUTION_WIDTH   = genDict['VERTEX_dZ_DISTRIBUTION_WIDTH']
    else:
        VERTEX_dZ_DISTRIBUTION_WIDTH   = 1.0
        
        
    np.random.seed(RSEED)
    
    nv_list=np.asarray(np.random.normal(PU_DISTRIBUTION_MEAN,PU_DISTRIBUTION_SIGMA,NEVENTS),dtype='int')
    for i in range(NEVENTS):
        while nv_list[i]<1:
            nv_list[i]=int(np.random.normal(PU_DISTRIBUTION_MEAN,PU_DISTRIBUTION_SIGMA))     


    nv=np.sum(nv_list)
    ntrk_list=np.asarray(np.random.normal(TRACKS_PER_VTX_MEAN,TRACKS_PER_VTX_SIGMA,nv),dtype='int')
    
    for i in range(nv):
        while ntrk_list[i]<1:
            ntrk_list[i]=int(np.random.normal(TRACKS_PER_VTX_MEAN,TRACKS_PER_VTX_SIGMA))  

    nt=np.sum(ntrk_list)
    track_dz=np.random.normal(TRACK_Z_DISTRIBUTION_BIAS,TRACK_Z_DISTRIBUTION_SIGMA,nt)
    
    if VERTEX_GEN_MODE=='Z':
        if VERTEX_Z_DISTRIBUTION_TYPE=='gaus':
            vtx_z  =np.random.normal(VERTEX_Z_DISTRIBUTION_MEAN,VERTEX_Z_DISTRIBUTION_SIGMA,nv)
        elif VERTEX_Z_DISTRIBUTION_TYPE=='uniform':
            minV = VERTEX_Z_DISTRIBUTION_MEAN - 0.5 * VERTEX_Z_DISTRIBUTION_WIDTH
            maxV = VERTEX_Z_DISTRIBUTION_MEAN + 0.5 * VERTEX_Z_DISTRIBUTION_WIDTH 
            vtx_z  =np.random.uniform(minV,maxV,nv)
    elif VERTEX_GEN_MODE=='dZ':
        vtx_z = getVtxWithDeltaZs(VERTEX_Z_DISTRIBUTION_MEAN,nv_list,VERTEX_dZ_DISTRIBUTION_MEAN,\
                                             VERTEX_dZ_DISTRIBUTION_WIDTH,VERTEX_dZ_DISTRIBUTION_TYPE)
        
    
    # print("track_dz shape = ", track_dz.shape,"( ",nt," )")
    # print("vtx_z shape   = ", vtx_z.shape,"( ",nv," )")

    track_z=track_dz*0.0
    k=0
    for i in range(nv):
        for j in range(ntrk_list[i]):
            track_z[k]=track_dz[k]+vtx_z[i]
            k+=1
            
    genData={}
    genData['track_z']  = track_z
    genData['track_dz'] = track_dz
    genData['vtx_z'] = vtx_z
    genData['nv_list']  = nv_list
    genData['nt_list']  = ntrk_list
    
    return genData

def plot_gen_distributions(nv_list,vtx_z,ntrk_list,track_dz):
    f=plt.figure(figsize=(15,10))
    ax0=plt.subplot2grid((2, 3), (0, 0), colspan=3)
    nph=ax0.hist(nv_list,bins=200,range=(0,200.0))
    ax0.set_title("PU disribution")

    ax1=plt.subplot2grid((2, 3), (1, 0), colspan=1)
    zvh=ax1.hist(vtx_z,bins=50)
    ax1.set_title("vtx z diatribution")

    ax2=plt.subplot2grid((2, 3), (1, 1), colspan=1)
    nth=ax2.hist(ntrk_list,bins=25)
    ax2.set_title("track assosiation to vtx")

    ax3=plt.subplot2grid((2, 3), (1, 2), colspan=1)
    zth=ax3.hist(track_dz,bins=40)
    ax3.set_title("track $\Delta$ z fromvtx")
    return f

def plot_tracks_for_vtxId(v_idx,ntrk_list,vtx_z,track_z):
    f=plt.figure(figsize=(5,5))
    i_s=np.sum(ntrk_list[0:v_idx])
    print("vertex centered around : ",vtx_z[v_idx])
    print("nearby vertex centers  : ",vtx_z[v_idx-1],vtx_z[v_idx+1])
    print("Number of tracks       : ",ntrk_list[v_idx])
    nph=plt.hist(track_z[i_s:i_s+ntrk_list[v_idx]],bins=5)
    print("tracks near boundaries :")
    print("\t",track_z[i_s-2:i_s+1])
    print("\t",track_z[i_s+ntrk_list[v_idx]-1:i_s+ntrk_list[v_idx]+2])
    plt.title("For vtx idx = "+str(v_idx))
    
def plot_event_details(evt_idx,nv_list,ntrk_list,vtx_z,track_z):
    i_s_v=np.sum(nv_list[0:evt_idx])
    i_f_v=i_s_v+nv_list[evt_idx]

    i_s_t=np.sum(ntrk_list[0:i_s_v])
    i_f_t=i_s_t+np.sum(ntrk_list[i_s_v:i_f_v])

    print("Number of vertices     : ",nv_list[evt_idx]," ( ",i_f_v-i_s_v," )")
    print("Number of tracks       : ",np.sum(ntrk_list[i_s_v:i_f_v]))
    f,ax=plt.subplots(1,3,figsize=(15,5))
    f.suptitle("For Event idx = "+str(evt_idx), fontsize=14)

    nph=ax[0].hist(vtx_z[i_s_v:i_f_v])
    ax[0].set_title("vtx z distribution")
    nph=ax[1].hist(ntrk_list[i_s_v:i_f_v])
    ax[1].set_title("track vtx assosiation distribution")
    nph=ax[2].hist(track_z[i_s_t:i_f_t])
    ax[2].set_title("track z distribution")