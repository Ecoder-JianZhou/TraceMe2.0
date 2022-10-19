from TraceMe_ioLibs import readNcFile
import numpy as np
from TraceMePlots import plotSpatialMap


class TraceMeBenchmarkAnalysis():
    def __init__(self):


    def runCLandBenchmark(self, nlatlon, lsObsFiles, ):
        """
            cland = cSoil + cVeg (kg C m-2)
            observed soil carbon use the HWSD with the circumpolar of NCSCDv2
        """
        latmin,latmax,lonmin,lonmax = nlatlon
        nlat, nlon = latmax-latmin, lonmax-lonmin

        for iFile in lsObsFiles:
            
        # filePath   = r"F:\ubuntu\Works\TraceMe_obs\obs_1d"
        filePath = r"C:\Users\jzhou\Documents\TraceMe_offline\inputData\obs_1d" ## bijiben
        # outPath  = r"F:\ubuntu\Works\TraceMe_figures\dataSources\model_results\TraceMe_workdir_historical"
        outPath  = r"C:\Users\jzhou\Documents\TraceMe_offline\figures\results\historical"

        ### observed cVeg 
        fileObsVeg = filePath+"/1_cVeg_obs_1kg.nc"
        datObsVeg  = readNcFile(fileObsVeg,"cVeg_obs")
        datObsVegLon = np.full((nlat,nlon), np.nan)
        datObsVegLon[:,:180] = datObsVeg[:,180:]
        datObsVegLon[:,180:] = datObsVeg[:,:180]

        ### observed cSoil: HWSD and NCSCDv2_Circumpolar
        filePathSoil   = filePath+"/5_cSoil"
        fileObsHWSD    = filePathSoil+"/HWSD_SOIL_CLM_RES_new_1deg.nc4"
        fileObsNCSCDv2 = filePathSoil+"/NCSCDv2_Circumpolar_WGS84_SOCC100_1deg.nc"
        ### Read data
        ### HWSD
        datObsHWSD     = readNcFile(fileObsHWSD,"AWT_SOC")
        datObsHWSDLon  = np.full((nlat,nlon), np.nan)
        datObsHWSDLon[:,:180] = datObsHWSD[:,180:]
        datObsHWSDLon[:,180:] = datObsHWSD[:,:180]
        ### NCSCDv2
        datObsNCSCDv2  = readNcFile(fileObsNCSCDv2,"NCSCDv2") ## lat: 34N to 90N
        datObsNCSCDv2  = datObsNCSCDv2[::-1,:]                ## turn up-down
        datNAN         = np.full((180-56,360),np.nan)         ## use NAN to fill the rest part
        datObsNCSCDv2  = np.vstack((datNAN,datObsNCSCDv2))
        ### if datObsNCSCD is nan, use the HWSD to replace.
        datObsMixCSoil = np.where(np.isnan(datObsNCSCDv2), datObsHWSDLon, datObsNCSCDv2)
        ### Total land Carbon
        datObsCLand = datObsMixCSoil + datObsVegLon

        ### Read model data ###
        # filePathModel = r"F:\ubuntu\Works\TraceMe_figures\dataSources\model_results\TraceMe_workdir_historical"
        filePathModel = r"C:\Users\jzhou\Documents\TraceMe_offline\figures\dataSources\model_results\TraceMe_workdir_historical"
        modelNames    = ['ACCESS-ESM1-5', 'BCC-CSM2-MR','CanESM5','CESM2',
                        'CNRM-ESM2-1', 'EC-Earth3-Veg','IPSL-CM6A-LR','MIROC-ES2L']
        varLongName   = ['baselineResidenceTime','cStorage','cStorageCap','cStoragePot','cue','environmentalScalars',
                        'gpp','npp','residenceTime','scalarPrecipitation','scalarTemperature']
        dataModel  = np.full((len(modelNames),nlat,nlon),np.nan)
        for indexModel, iModel in enumerate(modelNames):
            file = filePathModel+"/results_spatial_"+iModel+".nc"
            readData = readNcFile(file,'cStorage')
            dataModel[indexModel,:] = readData
        #### mean value ####
        dataMean     = np.mean(dataModel, axis=0)    # map is right
        dataMeanBias = (dataMean - datObsCLand)/datObsCLand  ## datObsCLand lon error ## ObsVegLon is right
        outFig = outPath + "/figure_bench_1_cland.png"
        titleName = "Bias: (model - observation)/observation"
        plotSpatialMap(nLatLon=[latmin,latmax,lonmin,lonmax],drawData=dataMeanBias,outFig=outFig,vMinMax=[-1,1],titleName=titleName,unit="kg C m-2")


############################################## 
#  cland = cSoil + cVeg (kg C m-2)
#
#  observed soil carbon use the HWSD with the circumpolar of NCSCDv2
#
##############################################

from TraceMe_ioLibs import readNcFile,mkColors
import numpy as np
import matplotlib.pyplot as plt

def plotLatitude(dat_x, dat_y, ls_modelName, colors, xylabel, outFig, dat_yy=None, figsize=[9,7]):       
    fontsize = 25
    fig = plt.figure(figsize=(figsize[0],figsize[1]))
    ax  = fig.add_axes([0.20,0.16,0.75,0.78])
    for i, modelName in enumerate(ls_modelName):
        plt.plot(dat_x[i,:], dat_y, colors[i], linewidth=2, label = modelName)
        if dat_yy is not None:
            plt.fill_between(x,dat_yy[i,:],dat_y[i,:],facecolor=colors[i],alpha=0.3)
    plt.yticks(np.linspace(-90,90,7))
    plt.ylim(-60,80)
    plt.xlabel(xylabel[0],fontsize=fontsize)
    plt.ylabel(xylabel[1],fontsize=fontsize)
    plt.tick_params(labelsize=22)
    #plt.legend(loc='right')#, bbox_to_anchor=(0,0),ncol=1, borderaxespad = 0.,fontsize=18)
    plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig(outFig, bbox_inches='tight')

latmin,latmax,lonmin,lonmax = -90,90,-180,180
nlat, nlon = latmax-latmin, lonmax-lonmin
# filePath   = r"F:\ubuntu\Works\TraceMe_obs\obs_1d"
filePath = r"C:\Users\jzhou\Documents\TraceMe_offline\inputData\obs_1d" ## bijiben
# outPath  = r"F:\ubuntu\Works\TraceMe_figures\dataSources\model_results\TraceMe_workdir_historical"
outPath  = r"C:\Users\jzhou\Documents\TraceMe_offline\figures\results\historical"

### observed cVeg 
fileObsVeg = filePath+"/1_cVeg_obs_1kg.nc"
datObsVeg  = readNcFile(fileObsVeg,"cVeg_obs")
datObsVegLon = np.full((nlat,nlon), np.nan)
datObsVegLon[:,:180] = datObsVeg[:,180:]
datObsVegLon[:,180:] = datObsVeg[:,:180]

### observed cSoil: HWSD and NCSCDv2_Circumpolar
filePathSoil   = filePath+"/5_cSoil"
fileObsHWSD    = filePathSoil+"/HWSD_SOIL_CLM_RES_new_1deg.nc4"
fileObsNCSCDv2 = filePathSoil+"/NCSCDv2_Circumpolar_WGS84_SOCC100_1deg.nc"
### Read data
### HWSD
datObsHWSD     = readNcFile(fileObsHWSD,"AWT_SOC")
datObsHWSDLon  = np.full((nlat,nlon), np.nan)
datObsHWSDLon[:,:180] = datObsHWSD[:,180:]
datObsHWSDLon[:,180:] = datObsHWSD[:,:180]
### NCSCDv2
datObsNCSCDv2  = readNcFile(fileObsNCSCDv2,"NCSCDv2") ## lat: 34N to 90N
datObsNCSCDv2  = datObsNCSCDv2[::-1,:]                ## turn up-down
datNAN         = np.full((180-56,360),np.nan)         ## use NAN to fill the rest part
datObsNCSCDv2  = np.vstack((datNAN,datObsNCSCDv2))
### if datObsNCSCD is nan, use the HWSD to replace.
datObsMixCSoil = np.where(np.isnan(datObsNCSCDv2), datObsHWSDLon, datObsNCSCDv2)
### Total land Carbon
datObsCLand = datObsMixCSoil + datObsVegLon

datObsClandLat = np.nanmean(datObsCLand,axis=1)

### Read model data ###
# filePathModel = r"F:\ubuntu\Works\TraceMe_figures\dataSources\model_results\TraceMe_workdir_historical"
filePathModel = r"C:\Users\jzhou\Documents\TraceMe_offline\figures\dataSources\model_results\TraceMe_workdir_historical"
modelNames    = ['ACCESS-ESM1-5', 'BCC-CSM2-MR','CanESM5','CESM2',
                 'CNRM-ESM2-1', 'EC-Earth3-Veg','IPSL-CM6A-LR','MIROC-ES2L']
varLongName   = ['baselineResidenceTime','cStorage','cStorageCap','cStoragePot','cue','environmentalScalars',
                 'gpp','npp','residenceTime','scalarPrecipitation','scalarTemperature']
dataModel  = np.full((len(modelNames),nlat,nlon),np.nan)
for indexModel, iModel in enumerate(modelNames):
    file = filePathModel+"/results_spatial_"+iModel+".nc"
    readData = readNcFile(file,'cStorage')
    dataModel[indexModel,:] = readData
#### mean value ####
dataMeanLat     = np.mean(dataModel, axis=2)    # map is right
# dataMeanBias = (dataMean - datObsCLand)/datObsCLand  ## datObsCLand lon error ## ObsVegLon is right
outFig = outPath + "/figure_bench_1_cland_lat.png"
titleName = "Latitude"
xyLabel = ["Carbon storage (kg C m-2)",'Latitude']
data4plot = np.full((9,nlat),np.nan)
data4plot[:8,:] = dataMeanLat 
data4plot[8,:] = datObsClandLat
colors = mkColors(8)
lsColors = []
lsColors = colors[1:]
lsColors.append(colors[0])
ls_LineNames = ['ACCESS-ESM1-5', 'BCC-CSM2-MR','CanESM5','CESM2',
                 'CNRM-ESM2-1', 'EC-Earth3-Veg','IPSL-CM6A-LR','MIROC-ES2L', 'Observation']
plotLatitude(dat_x        = data4plot,#[-90,90],
            dat_y        = np.arange(-90,90), #datSpt_gpp_lat,
            ls_modelName = ls_LineNames,
            colors       = lsColors,
            xylabel      = xyLabel,
            outFig       = outFig)





############################################## 
#  cland = cSoil + cVeg (kg C m-2)
#
#  observed soil carbon use the HWSD with the circumpolar of NCSCDv2
#
##############################################

from TraceMe_ioLibs import readNcFile,mkColors
import numpy as np
import matplotlib.pyplot as plt
from plotTaylor import plot_taylor,set_tayloraxes

def plotBenchTem(x, dat_y, xx, dat_yy, ls_modelName, ls_obsNames, colors, xylabel, outFig, titleName = None, figsize=[12,9]):      
    fontsize = 25
    fig = plt.figure(figsize=(figsize[0],figsize[1]))
    ax  = fig.add_axes([0.20,0.16,0.75,0.78])
    for i, modelName in enumerate(ls_modelName):
        plt.plot(x, dat_y[i,:], colors[i], linewidth=2, label = modelName)
    for j, obsName in enumerate(ls_obsNames):
        plt.plot(xx, dat_yy[j,:], colors[i+j+1], linewidth=2, label = obsName)
    ## whether add bar?
    # new_ticks = np.arange(range_x[0]-1,range_x[1])
    plt.xlim(np.min(x), np.max(x))
    # plt.xticks(new_ticks)
    plt.xlabel(xylabel[0],fontsize=fontsize)
    plt.ylabel(xylabel[1],fontsize=fontsize)
    plt.tick_params(labelsize=22)
    if titleName is not None:
        plt.title(titleName, fontsize=fontsize)
    #plt.legend(loc='right')#, bbox_to_anchor=(0,0),ncol=1, borderaxespad = 0.,fontsize=18)
    plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0., fontsize=18)
    plt.savefig(outFig, bbox_inches='tight')

def plotTaylor(datObs, datModels, ls_modelNames, colors, outFig,figsize=[12,9]):
    fig = plt.figure(figsize=(figsize[0],figsize[1]))
    ax  = set_tayloraxes(fig, 111) #fig.add_axes([0.20,0.16,0.75,0.78])       
    for index, iModel in enumerate(ls_modelNames):
        # print(datObs)
        # print(datModels[0,:])
        plot_taylor(axes= ax,refsample=datObs, sample=datModels[index,:],c=colors[index], marker="o")   
    plt.savefig(outFig, bbox_inches='tight')       

latmin,latmax,lonmin,lonmax = -90,90,-180,180
nlat, nlon = latmax-latmin, lonmax-lonmin
# filePath   = r"F:\ubuntu\Works\TraceMe_obs\obs_1d"
filePath = r"C:\Users\jzhou\Documents\TraceMe_offline\inputData\obs_1d" ## bijiben
# outPath  = r"F:\ubuntu\Works\TraceMe_figures\dataSources\model_results\TraceMe_workdir_historical"
outPath  = r"C:\Users\jzhou\Documents\TraceMe_offline\figures\results\historical"

### observed cVeg 
fileObsVeg = filePath+"/1_cVeg_obs_1kg.nc"
datObsVeg  = readNcFile(fileObsVeg,"cVeg_obs")
datObsVegLon = np.full((nlat,nlon), np.nan)
datObsVegLon[:,:180] = datObsVeg[:,180:]
datObsVegLon[:,180:] = datObsVeg[:,:180]

### observed cSoil: HWSD and NCSCDv2_Circumpolar
filePathSoil   = filePath+"/5_cSoil"
fileObsHWSD    = filePathSoil+"/HWSD_SOIL_CLM_RES_new_1deg.nc4"
fileObsNCSCDv2 = filePathSoil+"/NCSCDv2_Circumpolar_WGS84_SOCC100_1deg.nc"
### Read data
### HWSD
datObsHWSD     = readNcFile(fileObsHWSD,"AWT_SOC")
datObsHWSDLon  = np.full((nlat,nlon), np.nan)
datObsHWSDLon[:,:180] = datObsHWSD[:,180:]
datObsHWSDLon[:,180:] = datObsHWSD[:,:180]
### NCSCDv2
datObsNCSCDv2  = readNcFile(fileObsNCSCDv2,"NCSCDv2") ## lat: 34N to 90N
datObsNCSCDv2  = datObsNCSCDv2[::-1,:]                ## turn up-down
datNAN         = np.full((180-56,360),np.nan)         ## use NAN to fill the rest part
datObsNCSCDv2  = np.vstack((datNAN,datObsNCSCDv2))
### if datObsNCSCD is nan, use the HWSD to replace.
datObsMixCSoil = np.where(np.isnan(datObsNCSCDv2), datObsHWSDLon, datObsNCSCDv2)
### Total land Carbon
datObsCLand = datObsMixCSoil + datObsVegLon
### Global total land carbon
dat_area = readNcFile("area.nc","area")*1e6
datObsCLandTem = np.nansum(datObsCLand*dat_area)/1e12

### Read model data ###
# filePathModel = r"F:\ubuntu\Works\TraceMe_figures\dataSources\model_results\TraceMe_workdir_historical"
filePathModel = r"C:\Users\jzhou\Documents\TraceMe_offline\figures\dataSources\model_results\TraceMe_workdir_historical"
modelNames    = ['ACCESS-ESM1-5', 'BCC-CSM2-MR','CanESM5','CESM2',
                 'CNRM-ESM2-1', 'EC-Earth3-Veg','IPSL-CM6A-LR','MIROC-ES2L']
varLongName   = ['baselineResidenceTime','cStorage','cStorageCap','cStoragePot','cue','environmentalScalars',
                 'gpp','npp','residenceTime','scalarPrecipitation','scalarTemperature']
ntime = 164
dataModel  = np.full((len(modelNames),ntime),np.nan)
for indexModel, iModel in enumerate(modelNames):
    file = filePathModel+"/results_temporal_"+iModel+".nc"
    readData = readNcFile(file,'cStorage')
    dataModel[indexModel,:] = readData
colors = mkColors(8)
lsColors = []
lsColors = colors[1:]
lsColors.append(colors[0])
print(lsColors)
outFig  = outPath + "/figure_bench_1_landc_temporal_taylor.png"
dat_yy = np.zeros((1,ntime))+datObsCLandTem
# dataModel = dataModel*0 + datObsCLandTem
a, b = np.min(dat_yy), np.min(dataModel)
c, d = np.max(dat_yy), np.max(dataModel)
datMin = np.min([a,b])
datMax = np.max([np.max(dat_yy),np.max(dataModel)])
dat_yy = (dat_yy - datMin)/(datMax - datMin)
dataModel = (dataModel - datMin)/(datMax-datMin)
plotTaylor(datObs=dat_yy,datModels=dataModel,ls_modelNames=modelNames,colors=colors[1:],outFig=outFig)
exit(0)
xylabel = ["time (year)", "Land Carbon Storage (Pg C)"]
outFig = outPath + "/figure_bench_1_landc_temporal.png"
plotBenchTem(x  = np.linspace(1851,2014,164), dat_y = dataModel,
             xx = np.linspace(1851,2014,164), dat_yy = np.zeros((1,ntime))+datObsCLandTem,
             ls_modelName=modelNames,ls_obsNames=["Observation"],colors=lsColors,
             xylabel=xylabel,outFig=outFig)
exit(0)
#### mean value ####
dataMean     = np.mean(dataModel, axis=0)    # map is right
dataMeanBias = (dataMean - datObsCLand)/datObsCLand  ## datObsCLand lon error ## ObsVegLon is right
outFig = outPath + "/figure_bench_1_cland.png"
titleName = "Bias: (model - observation)/observation"



############################################## 
# Observed NPP: 1. CRUNCEP_V4P1; 2. CRUNCEPv8_V4P1;
#               3. V4;           4. MERRA2_V4;
#               5. NCEPR2_V4;    6. MODIS.
#
# Evaluation: 1. each observed NPP vs mean of models
#             2. mean observed NPP vs mean of models
# 
# Map: 1. mean of all observed NPP.
#      2. mean of all modeled NPP.
#      3. Bias: (modeled NPP - observed NPP) / observed NPP
#==============================
# bug: 20210703: model results have no temperol global data
##############################################

from TraceMe_ioLibs import readNcFile,convertLon
import numpy as np
from TraceMePlots import plotSpatialMap

def convertLonWithTime(data, nLatLon):
    lenTime = data.shape[0]
    returnData = np.full((data.shape[0],data.shape[1],data.shape[2]), np.nan)
    for iTime in range(lenTime):
        returnData[iTime,:] = convertLon(data[iTime,:],nLatLon)
    return returnData

latmin,latmax,lonmin,lonmax = -90,90,-180,180
nLatLon=[latmin,latmax,lonmin,lonmax]
nlat, nlon = latmax-latmin, lonmax-lonmin
# filePath   = r"F:\ubuntu\Works\TraceMe_obs\obs_1d"
filePath = r"C:\Users\jzhou\Documents\TraceMe_offline\inputData\obs_1d" ## bijiben
# outPath  = r"F:\ubuntu\Works\TraceMe_figures\dataSources\model_results\TraceMe_workdir_historical"
outPath  = r"C:\Users\jzhou\Documents\TraceMe_offline\figures\results\historical"

### observed NPP 1
lsObsNPP = ["2_1_npp_CRUNCEP_V4P1_Standard_1982_2016_Annual_GEO_30min_1d.nc",
            "2_2_npp_CRUNCEPv8_V4P1_Standard_1982_2016_Annual_GEO_30min_1d.nc",
            "2_3_npp_V4_Standard_1982_2015_Annual_GEO_30min_1d.nc",
            "2_4_npp_MERRA2_V4_Standard_1982_2015_Annual_GEO_30min_1d.nc",
            "2_5_npp_NCEPR2_V4_Standard_1982_2015_Annual_GEO_30min_1d.nc",
            "2_6_npp_modis_obs_1d_2000_2014.nc"]
lsObsNPPNames = ["NPP","NPP","NPP","NPP","NPP","npp_modis"]
startYear = [1982,1982,1982,1982,1982,2000]
endYear   = [2016,2016,2015,2015,2015,2014]
lenTime   = [35,  35,  34,  34,  34,  15]

datObsNPP_CRU   = convertLonWithTime(readNcFile(filePath+"/"+lsObsNPP[0],lsObsNPPNames[0]),nLatLon)/1000 # to kg m-2  #np.full((lenTime[0],len(nlat),len(nlon)),np.nan)
datObsNPP_CRUv8 = convertLonWithTime(readNcFile(filePath+"/"+lsObsNPP[1],lsObsNPPNames[1]),nLatLon)/1000 # to kg m-2 #np.full((lenTime[1],len(nlat),len(nlon)),np.nan)
datObsNPP_V4    = convertLonWithTime(readNcFile(filePath+"/"+lsObsNPP[2],lsObsNPPNames[2]),nLatLon)/1000 # to kg m-2 #np.full((lenTime[2],len(nlat),len(nlon)),np.nan)
datObsNPP_MER   = convertLonWithTime(readNcFile(filePath+"/"+lsObsNPP[3],lsObsNPPNames[3]),nLatLon)/1000 # to kg m-2 #np.full((lenTime[3],len(nlat),len(nlon)),np.nan)
datObsNPP_NCE   = convertLonWithTime(readNcFile(filePath+"/"+lsObsNPP[4],lsObsNPPNames[4]),nLatLon)/1000 # to kg m-2 #np.full((lenTime[4],len(nlat),len(nlon)),np.nan)
datObsNPP_MODIS = convertLonWithTime(readNcFile(filePath+"/"+lsObsNPP[5],lsObsNPPNames[5]),nLatLon)/1000 # to kg m-2 #np.full((lenTime[5],len(nlat),len(nlon)),np.nan)

datObsNPPMean = np.full((6,nlat,nlon),np.nan)
datObsNPPMean[0,:] = np.mean(datObsNPP_CRU, axis=0)
datObsNPPMean[1,:] = np.mean(datObsNPP_CRUv8, axis=0)
datObsNPPMean[2,:] = np.mean(datObsNPP_V4, axis=0)
datObsNPPMean[3,:] = np.mean(datObsNPP_MER, axis=0)
datObsNPPMean[4,:] = np.mean(datObsNPP_NCE, axis=0)
datObsNPPMean[5,:] = np.mean(datObsNPP_MODIS, axis=0)
############################ Global map ############################
obsNPPNames = ["CRUNCEP_V4P1", "CRUNCEPv8_V4P1", "V4", "MERRA2","NCEPR2","MODIS"]
vMinMax     = [0,3]
unit="kg C m-2"
for index, iName in enumerate(obsNPPNames):
    outFig = outPath+"/"+"figure_bench_2_npp_obs_"+iName+".png"
    titleName = "The observed NPP ("+iName +": "+str(startYear[index])+"-" + str(endYear[index]) + ")"
    plotSpatialMap(nLatLon = nLatLon, drawData=datObsNPPMean[index],outFig = outFig,vMinMax=vMinMax,titleName=titleName, unit=unit)

####### model ######
### Read model data ###
# filePathModel = r"F:\ubuntu\Works\TraceMe_figures\dataSources\model_results\TraceMe_workdir_historical"
filePathModel = r"C:\Users\jzhou\Documents\TraceMe_offline\figures\dataSources\model_results\TraceMe_workdir_historical"
modelNames    = ['ACCESS-ESM1-5', 'BCC-CSM2-MR','CanESM5','CESM2',
                 'CNRM-ESM2-1', 'EC-Earth3-Veg','IPSL-CM6A-LR','MIROC-ES2L']
varLongName   = ['baselineResidenceTime','cStorage','cStorageCap','cStoragePot','cue','environmentalScalars',
                 'gpp','npp','residenceTime','scalarPrecipitation','scalarTemperature']
dataModel  = np.full((len(modelNames),nlat,nlon),np.nan)
for indexModel, iModel in enumerate(modelNames):
    file = filePathModel+"/results_spatial_"+iModel+".nc"
    readData = readNcFile(file,'npp')
    dataModel[indexModel,:] = readData
#     plotSpatialMap(nLatLon=nLatLon,drawData=readData,outFig=outPath+"/figure_bench_2_npp_model_"+iModel+".png",
#                    vMinMax=vMinMax,titleName="NPP ("+iModel+")",unit="kg C m-2")
########### Bias ########
### obs mean ###
datObsNPPAllMean = np.nanmean(datObsNPPMean,axis=0)
# outFig = outPath+"/figure_bench_2_npp_obsAllMean.png"
# titleName = "NPP (the mean of all observations)"
# plotSpatialMap(nLatLon=nLatLon,drawData=datObsNPPAllMean,outFig=outFig,vMinMax=vMinMax,titleName=titleName,unit="kg C m-2")
### model mean ###
datModelAllMean = np.mean(dataModel, axis=0)
# outFig = outPath+"/figure_bench_2_npp_modelAllMean.png"
# titleName = "NPP (the mean of all models)"
# plotSpatialMap(nLatLon=nLatLon,drawData=datModelAllMean,outFig=outFig,vMinMax=vMinMax,titleName=titleName,unit="kg C m-2")
## Bias ###
datBiasNPP = (datModelAllMean - datObsNPPAllMean)/datObsNPPAllMean
outFig = outPath+"/figure_bench_2_npp_bias.png"
titleName = "NPP (Bias)"
plotSpatialMap(nLatLon=nLatLon,drawData=datBiasNPP,outFig=outFig,vMinMax=[-1,1],titleName=titleName,unit="kg C m-2",cmap="bwr")
exit(0)

############################################## 
# Observed GPP: 1. CRUNCEP_V4P1; 2. CRUNCEPv8_V4P1;
#               3. V4;           4. MERRA2_V4;
#               5. NCEPR2_V4;    6. MODIS;
#               7. FLUXCOMGPP;   8. GOSIF;
#               9. VPM;          10. MTE.
#
# Evaluation: 1. each observed GPP vs mean of models
#             2. mean observed GPP vs mean of models
# 
# Map: 1. mean of all observed GPP, include the average of all observations.
#      2. mean of all modeled GPP, include the average of all models.
#      3. Bias: (modeled GPP - observed GPP) / observed GPP
#==============================
# bug: 20210704: model results have no temperol global distribution data
##############################################

from TraceMe_ioLibs import readNcFile,convertLon
import numpy as np
from TraceMePlots import plotSpatialMap

def convertLonWithTime(data, nLatLon):
    lenTime = data.shape[0]
    returnData = np.full((data.shape[0],data.shape[1],data.shape[2]), np.nan)
    for iTime in range(lenTime):
        returnData[iTime,:] = convertLon(data[iTime,:],nLatLon)
    return returnData

latmin,latmax,lonmin,lonmax = -90,90,-180,180
nLatLon=[latmin,latmax,lonmin,lonmax]
nlat, nlon = latmax-latmin, lonmax-lonmin
# filePath   = r"F:\ubuntu\Works\TraceMe_obs\obs_1d"
filePath = r"C:\Users\jzhou\Documents\TraceMe_offline\inputData\obs_1d" ## bijiben
# outPath  = r"F:\ubuntu\Works\TraceMe_figures\dataSources\model_results\TraceMe_workdir_historical"
outPath  = r"C:\Users\jzhou\Documents\TraceMe_offline\figures\results\historical"

### observed GPP 1
lsObsGPP = ["2_1_gpp_CRUNCEP_V4P1_Standard_1982_2016_Annual_GEO_30min_1d.nc",
            "2_2_gpp_CRUNCEPv8_V4P1_Standard_1982_2016_Annual_GEO_30min_1d.nc",
            "2_3_gpp_V4_Standard_1982_2015_Annual_GEO_30min_1d.nc",
            "2_4_gpp_MERRA2_V4_Standard_1982_2015_Annual_GEO_30min_1d.nc",
            "2_5_gpp_NCEPR2_V4_Standard_1982_2015_Annual_GEO_30min_1d.nc",
            "3_3_gpp_MODIS_1kg_new123.nc",
            "3_1_gpp_FLUXCOMGPP_Annual1980-2013_1d_year.nc",
            "3_2_gpp_GOSIF_1kg_new.nc",
            "3_4_gpp_VPM_1kg_new.nc",
            "3_5_gpp_MTE_AnnualGPP_1982-2011_1d_yearsum1000.nc"]
lsObsGPPNames = ["GPP","GPP","GPP","GPP","GPP","gpp","GPP","gpp","gpp","gpp"]
startYear = [1982,1982,1982,1982,1982,2000,1980,2001,2000,1982]
endYear   = [2016,2016,2015,2015,2015,2014,2013,2014,2014,2011]
lenTime   = [35,  35,  34,  34,  34,  15,  34,  14,  15,  30]

datObsGPP_CRU   = convertLonWithTime(readNcFile(filePath+"/"+lsObsGPP[0],lsObsGPPNames[0]),nLatLon)/1000 # to kg m-2  #np.full((lenTime[0],len(nlat),len(nlon)),np.nan)
datObsGPP_CRUv8 = convertLonWithTime(readNcFile(filePath+"/"+lsObsGPP[1],lsObsGPPNames[1]),nLatLon)/1000 # to kg m-2 #np.full((lenTime[1],len(nlat),len(nlon)),np.nan)
datObsGPP_V4    = convertLonWithTime(readNcFile(filePath+"/"+lsObsGPP[2],lsObsGPPNames[2]),nLatLon)/1000 # to kg m-2 #np.full((lenTime[2],len(nlat),len(nlon)),np.nan)
datObsGPP_MER   = convertLonWithTime(readNcFile(filePath+"/"+lsObsGPP[3],lsObsGPPNames[3]),nLatLon)/1000 # to kg m-2 #np.full((lenTime[3],len(nlat),len(nlon)),np.nan)
datObsGPP_NCE   = convertLonWithTime(readNcFile(filePath+"/"+lsObsGPP[4],lsObsGPPNames[4]),nLatLon)/1000 # to kg m-2 #np.full((lenTime[4],len(nlat),len(nlon)),np.nan)
datObsGPP_MODIS = convertLonWithTime(readNcFile(filePath+"/"+lsObsGPP[5],lsObsGPPNames[5]),nLatLon)/1000 # to kg m-2 #np.full((lenTime[5],len(nlat),len(nlon)),np.nan)
datObsGPP_FLUX  = convertLonWithTime(readNcFile(filePath+"/"+lsObsGPP[6],lsObsGPPNames[6]),nLatLon)/1000 # to kg m-2 #np.full((lenTime[2],len(nlat),len(nlon)),np.nan)
datObsGPP_GOSIF = convertLonWithTime(readNcFile(filePath+"/"+lsObsGPP[7],lsObsGPPNames[7]),nLatLon)/1000 # to kg m-2 #np.full((lenTime[3],len(nlat),len(nlon)),np.nan)
datObsGPP_VPM   = convertLonWithTime(readNcFile(filePath+"/"+lsObsGPP[8],lsObsGPPNames[8]),nLatLon)/1000 # to kg m-2 #np.full((lenTime[4],len(nlat),len(nlon)),np.nan)
datObsGPP_MTE   = convertLonWithTime(readNcFile(filePath+"/"+lsObsGPP[9],lsObsGPPNames[9]),nLatLon)/1000 # to kg m-2 #np.full((lenTime[5],len(nlat),len(nlon)),np.nan)

datObsGPPMean = np.full((len(lsObsGPP),nlat,nlon),np.nan)
datObsGPPMean[0,:] = np.mean(datObsGPP_CRU, axis=0)
datObsGPPMean[1,:] = np.mean(datObsGPP_CRUv8, axis=0)
datObsGPPMean[2,:] = np.mean(datObsGPP_V4, axis=0)
datObsGPPMean[3,:] = np.mean(datObsGPP_MER, axis=0)
datObsGPPMean[4,:] = np.mean(datObsGPP_NCE, axis=0)
datObsGPPMean[5,:] = np.mean(datObsGPP_MODIS, axis=0)
datObsGPPMean[6,:] = np.mean(datObsGPP_FLUX, axis=0)
datObsGPPMean[7,:] = np.mean(datObsGPP_GOSIF, axis=0)
datObsGPPMean[8,:] = np.mean(datObsGPP_VPM, axis=0)
datObsGPPMean[9,:] = np.mean(datObsGPP_MTE, axis=0)
############################ Global map ############################
obsGPPNames = ["CRUNCEP_V4P1", "CRUNCEPv8_V4P1", "V4", "MERRA2","NCEPR2","MODIS","FLUXCOMGPP","GOSIF","VPM","MTE"]
vMinMax     = [0,5]
unit="kg C m-2"
for index, iName in enumerate(obsGPPNames):
    outFig = outPath+"/"+"figure_bench_3_GPP_obs_"+iName+".png"
    titleName = "The observed GPP ("+iName +": "+str(startYear[index])+"-" + str(endYear[index]) + ")"
    plotSpatialMap(nLatLon = nLatLon, drawData=datObsGPPMean[index],outFig = outFig,vMinMax=vMinMax,titleName=titleName, unit=unit)

####### model ######
### Read model data ###
# filePathModel = r"F:\ubuntu\Works\TraceMe_figures\dataSources\model_results\TraceMe_workdir_historical"
filePathModel = r"C:\Users\jzhou\Documents\TraceMe_offline\figures\dataSources\model_results\TraceMe_workdir_historical"
modelNames    = ['ACCESS-ESM1-5', 'BCC-CSM2-MR','CanESM5','CESM2',
                 'CNRM-ESM2-1', 'EC-Earth3-Veg','IPSL-CM6A-LR','MIROC-ES2L']
varLongName   = ['baselineResidenceTime','cStorage','cStorageCap','cStoragePot','cue','environmentalScalars',
                 'gpp','npp','residenceTime','scalarPrecipitation','scalarTemperature']
dataModel  = np.full((len(modelNames),nlat,nlon),np.nan)
for indexModel, iModel in enumerate(modelNames):
    file = filePathModel+"/results_spatial_"+iModel+".nc"
    readData = readNcFile(file,'gpp')
    dataModel[indexModel,:] = readData
    # plotSpatialMap(nLatLon=nLatLon,drawData=readData,outFig=outPath+"/figure_bench_3_GPP_model_"+iModel+".png",
    #                vMinMax=vMinMax,titleName="GPP ("+iModel+")",unit="kg C m-2")
########### Bias ########
### obs mean ###
datObsGPPAllMean = np.nanmean(datObsGPPMean,axis=0)
outFig = outPath+"/figure_bench_3_GPP_obsAllMean.png"
titleName = "GPP (the mean of all observations)"
# plotSpatialMap(nLatLon=nLatLon,drawData=datObsGPPAllMean,outFig=outFig,vMinMax=vMinMax,titleName=titleName,unit="kg C m-2")
### model mean ###
datModelAllMean = np.mean(dataModel, axis=0)
outFig = outPath+"/figure_bench_3_GPP_modelAllMean.png"
titleName = "GPP (the mean of all models)"
# plotSpatialMap(nLatLon=nLatLon,drawData=datModelAllMean,outFig=outFig,vMinMax=vMinMax,titleName=titleName,unit="kg C m-2")
## Bias ###
datBiasGPP = (datModelAllMean - datObsGPPAllMean)/datObsGPPAllMean
outFig = outPath+"/figure_bench_3_GPP_bias.png"
titleName = "GPP (Bias)"
plotSpatialMap(nLatLon=nLatLon,drawData=datBiasGPP,outFig=outFig,vMinMax=[-1,1],titleName=titleName,unit="kg C m-2",cmap="bwr")


############################################## 
# Observed tau: 1. WholeEco; 2. LandGIS; 
#               3. S2017;    4. SoilGrids;
#
# Evaluation: 1. each observed tau vs mean of models
#             2. mean observed tau vs mean of models
# 
# Map: 1. mean of all observed tau, include the average of all observations.
#      2. mean of all modeled tau, include the average of all models.
#      3. Bias: (modeled tau - observed tau) / observed tau
#==============================
# bug: 20210704: model results have no temperol global distribution data
##############################################

from TraceMe_ioLibs import readNcFile,convertLon
import numpy as np
from TraceMePlots import plotSpatialMap

def convertLonWithTime(data, nLatLon):
    lenTime = data.shape[0]
    returnData = np.full((data.shape[0],data.shape[1],data.shape[2]), np.nan)
    for iTime in range(lenTime):
        returnData[iTime,:] = convertLon(data[iTime,:],nLatLon)
    return returnData

latmin,latmax,lonmin,lonmax = -90,90,-180,180
nLatLon=[latmin,latmax,lonmin,lonmax]
nlat, nlon = latmax-latmin, lonmax-lonmin
# filePath   = r"F:\ubuntu\Works\TraceMe_obs\obs_1d"
filePath = r"C:\Users\jzhou\Documents\TraceMe_offline\inputData\obs_1d" ## bijiben
# outPath  = r"F:\ubuntu\Works\TraceMe_figures\dataSources\model_results\TraceMe_workdir_historical"
outPath  = r"C:\Users\jzhou\Documents\TraceMe_offline\figures\results\historical"

### observed tau 1
lsObsTau = ["4_1_tau_all_1d.nc",
            "4_2_tau_LandGIS_1d.nc",
            "4_3_tau_S2017_1d.nc",
            "4_4_tau_SoilGrids_1d.nc"]
lsObsTauNames = ["tau","tau","tau","tau"]
# startYear = [1982,1982,1982,1982,1982,2000,1980,2001,2000,1982]
# endYear   = [2016,2016,2015,2015,2015,2014,2013,2014,2014,2011]
# lenTime   = [35,  35,  34,  34,  34,  15,  34,  14,  15,  30]

datObsTauMean = np.full((len(lsObsTau),nlat,nlon),np.nan)
datObsTauMean[0,:] = convertLonWithTime(readNcFile(filePath+"/"+lsObsTau[0],lsObsTauNames[0]),nLatLon)[0,:] # 1m
datObsTauMean[1,:] = convertLonWithTime(readNcFile(filePath+"/"+lsObsTau[1],lsObsTauNames[1]),nLatLon)[0,:] # 1m
datObsTauMean[2,:] = convertLonWithTime(readNcFile(filePath+"/"+lsObsTau[2],lsObsTauNames[2]),nLatLon)[0,:] # 1m
datObsTauMean[3,:] = convertLonWithTime(readNcFile(filePath+"/"+lsObsTau[3],lsObsTauNames[3]),nLatLon)[0,:] # 1m

############################ Global map ############################
obsTauNames = ["All_Ecosystem", "LandGIS", "S2017", "SoilGrids"]
vMinMax     = [0,1000]
unit="kg C m-2"
for index, iName in enumerate(obsTauNames):
    outFig = outPath+"/"+"figure_bench_4_Tau_obs_"+iName+".png"
    titleName = "The observed tau ("+iName  + ")"
    plotSpatialMap(nLatLon = nLatLon, drawData=datObsTauMean[index],outFig = outFig,vMinMax=vMinMax,titleName=titleName, unit=unit)

####### model ######
### Read model data ###
# filePathModel = r"F:\ubuntu\Works\TraceMe_figures\dataSources\model_results\TraceMe_workdir_historical"
filePathModel = r"C:\Users\jzhou\Documents\TraceMe_offline\figures\dataSources\model_results\TraceMe_workdir_historical"
modelNames    = ['ACCESS-ESM1-5', 'BCC-CSM2-MR','CanESM5','CESM2',
                 'CNRM-ESM2-1', 'EC-Earth3-Veg','IPSL-CM6A-LR','MIROC-ES2L']
varLongName   = ['baselineResidenceTime','cStorage','cStorageCap','cStoragePot','cue','environmentalScalars',
                 'gpp','npp','residenceTime','scalarPrecipitation','scalarTemperature']
dataModel  = np.full((len(modelNames),nlat,nlon),np.nan)
for indexModel, iModel in enumerate(modelNames):
    file = filePathModel+"/results_spatial_"+iModel+".nc"
    readData = readNcFile(file,'residenceTime')
    dataModel[indexModel,:] = readData
    plotSpatialMap(nLatLon=nLatLon,drawData=readData,outFig=outPath+"/figure_bench_4_Tau_model_"+iModel+".png",
                   vMinMax=vMinMax,titleName="Tau ("+iModel+")",unit="year")
########### Bias ########
### obs mean ###
datObsTauAllMean = np.nanmean(datObsTauMean,axis=0)
outFig = outPath+"/figure_bench_4_tau_obsAllMean.png"
titleName = "Tau (the mean of all observations)"
plotSpatialMap(nLatLon=nLatLon,drawData=datObsTauAllMean,outFig=outFig,vMinMax=vMinMax,titleName=titleName,unit="year")
### model mean ###
datModelAllMean = np.mean(dataModel, axis=0)
outFig = outPath+"/figure_bench_4_tau_modelAllMean.png"
titleName = "Tau (the mean of all models)"
plotSpatialMap(nLatLon=nLatLon,drawData=datModelAllMean,outFig=outFig,vMinMax=vMinMax,titleName=titleName,unit="year")
## Bias ###
datBiasTau = (datModelAllMean - datObsTauAllMean)/datObsTauAllMean
outFig = outPath+"/figure_bench_4_tau_bias.png"
titleName = "Tau (Bias)"
plotSpatialMap(nLatLon=nLatLon,drawData=datBiasTau,outFig=outFig,vMinMax=[-1,1],titleName=titleName,unit="year",cmap="bwr")


############################################## 
# Observed tau: 1. WholeEco; 2. LandGIS; 
#               3. S2017;    4. SoilGrids;
#
# Evaluation: 1. each observed tau vs mean of models
#             2. mean observed tau vs mean of models
# 
# Map: 1. mean of all observed tau, include the average of all observations.
#      2. mean of all modeled tau, include the average of all models.
#      3. Bias: (modeled tau - observed tau) / observed tau
#==============================
# bug: 20210704: model results have no temperol global distribution data
##############################################

from TraceMe_ioLibs import readNcFile,convertLon
import numpy as np
from TraceMePlots import plotSpatialMap

def convertLonWithTime(data, nLatLon):
    lenTime = data.shape[0]
    returnData = np.full((data.shape[0],data.shape[1],data.shape[2]), np.nan)
    for iTime in range(lenTime):
        returnData[iTime,:] = convertLon(data[iTime,:],nLatLon)
    return returnData

latmin,latmax,lonmin,lonmax = -90,90,-180,180
nLatLon=[latmin,latmax,lonmin,lonmax]
nlat, nlon = latmax-latmin, lonmax-lonmin
# filePath   = r"F:\ubuntu\Works\TraceMe_obs\obs_1d"
filePath = r"C:\Users\jzhou\Documents\TraceMe_offline\inputData\obs_1d" ## bijiben
# outPath  = r"F:\ubuntu\Works\TraceMe_figures\dataSources\model_results\TraceMe_workdir_historical"
outPath  = r"C:\Users\jzhou\Documents\TraceMe_offline\figures\results\historical"

### observed tau 1
lsObsTau = ["4_1_tau_all_1d.nc",
            "4_2_tau_LandGIS_1d.nc",
            "4_3_tau_S2017_1d.nc",
            "4_4_tau_SoilGrids_1d.nc"]
lsObsTauNames = ["tau","tau","tau","tau"]
# startYear = [1982,1982,1982,1982,1982,2000,1980,2001,2000,1982]
# endYear   = [2016,2016,2015,2015,2015,2014,2013,2014,2014,2011]
# lenTime   = [35,  35,  34,  34,  34,  15,  34,  14,  15,  30]

datObsTauMean = np.full((len(lsObsTau),nlat,nlon),np.nan)
datObsTauMean[0,:] = convertLonWithTime(readNcFile(filePath+"/"+lsObsTau[0],lsObsTauNames[0]),nLatLon)[0,:] # 1m
datObsTauMean[1,:] = convertLonWithTime(readNcFile(filePath+"/"+lsObsTau[1],lsObsTauNames[1]),nLatLon)[0,:] # 1m
datObsTauMean[2,:] = convertLonWithTime(readNcFile(filePath+"/"+lsObsTau[2],lsObsTauNames[2]),nLatLon)[0,:] # 1m
datObsTauMean[3,:] = convertLonWithTime(readNcFile(filePath+"/"+lsObsTau[3],lsObsTauNames[3]),nLatLon)[0,:] # 1m

############################ Global map ############################
obsTauNames = ["All_Ecosystem", "LandGIS", "S2017", "SoilGrids"]
vMinMax     = [0,1000]
unit="kg C m-2"
for index, iName in enumerate(obsTauNames):
    outFig = outPath+"/"+"figure_bench_4_Tau_obs_"+iName+".png"
    titleName = "The observed tau ("+iName  + ")"
    plotSpatialMap(nLatLon = nLatLon, drawData=datObsTauMean[index],outFig = outFig,vMinMax=vMinMax,titleName=titleName, unit=unit)

####### model ######
### Read model data ###
# filePathModel = r"F:\ubuntu\Works\TraceMe_figures\dataSources\model_results\TraceMe_workdir_historical"
filePathModel = r"C:\Users\jzhou\Documents\TraceMe_offline\figures\dataSources\model_results\TraceMe_workdir_historical"
modelNames    = ['ACCESS-ESM1-5', 'BCC-CSM2-MR','CanESM5','CESM2',
                 'CNRM-ESM2-1', 'EC-Earth3-Veg','IPSL-CM6A-LR','MIROC-ES2L']
varLongName   = ['baselineResidenceTime','cStorage','cStorageCap','cStoragePot','cue','environmentalScalars',
                 'gpp','npp','residenceTime','scalarPrecipitation','scalarTemperature']
dataModel  = np.full((len(modelNames),nlat,nlon),np.nan)
for indexModel, iModel in enumerate(modelNames):
    file = filePathModel+"/results_spatial_"+iModel+".nc"
    readData = readNcFile(file,'residenceTime')
    dataModel[indexModel,:] = readData
    plotSpatialMap(nLatLon=nLatLon,drawData=readData,outFig=outPath+"/figure_bench_4_Tau_model_"+iModel+".png",
                   vMinMax=vMinMax,titleName="Tau ("+iModel+")",unit="year")
########### Bias ########
### obs mean ###
datObsTauAllMean = np.nanmean(datObsTauMean,axis=0)
outFig = outPath+"/figure_bench_4_tau_obsAllMean.png"
titleName = "Tau (the mean of all observations)"
plotSpatialMap(nLatLon=nLatLon,drawData=datObsTauAllMean,outFig=outFig,vMinMax=vMinMax,titleName=titleName,unit="year")
### model mean ###
datModelAllMean = np.mean(dataModel, axis=0)
outFig = outPath+"/figure_bench_4_tau_modelAllMean.png"
titleName = "Tau (the mean of all models)"
plotSpatialMap(nLatLon=nLatLon,drawData=datModelAllMean,outFig=outFig,vMinMax=vMinMax,titleName=titleName,unit="year")
## Bias ###
datBiasTau = (datModelAllMean - datObsTauAllMean)/datObsTauAllMean
outFig = outPath+"/figure_bench_4_tau_bias.png"
titleName = "Tau (Bias)"
plotSpatialMap(nLatLon=nLatLon,drawData=datBiasTau,outFig=outFig,vMinMax=[-1,1],titleName=titleName,unit="year",cmap="bwr")





from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import floating_axes
from mpl_toolkits.axisartist import grid_finder
import numpy as np
import matplotlib.pyplot as plt

def set_tayloraxes(fig, location=111):
    trans = PolarAxes.PolarTransform()
    r1_locs = np.hstack((np.arange(1,10)/10.0,[0.95,0.99]))
    t1_locs = np.arccos(r1_locs)        
    gl1 = grid_finder.FixedLocator(t1_locs)    
    tf1 = grid_finder.DictFormatter(dict(zip(t1_locs, map(str,r1_locs))))
    r2_locs = np.arange(0,2,0.25)
    r2_labels = ['0 ', '0.25 ', '0.50 ', '0.75 ', 'REF ', '1.25 ', '1.50 ', '1.75 ']
    gl2 = grid_finder.FixedLocator(r2_locs)
    tf2 = grid_finder.DictFormatter(dict(zip(r2_locs, map(str,r2_labels))))
    ghelper = floating_axes.GridHelperCurveLinear(trans,extremes=(0,np.pi/2,0,1.75),
                                                  grid_locator1=gl1,tick_formatter1=tf1,
                                                  grid_locator2=gl2,tick_formatter2=tf2)
    ax = floating_axes.FloatingSubplot(fig, location, grid_helper=ghelper)
    fig.add_subplot(ax)
    ax.axis["top"].set_axis_direction("bottom")  
    ax.axis["top"].toggle(ticklabels=True, label=True)
    ax.axis["top"].major_ticklabels.set_axis_direction("top")
    ax.axis["top"].label.set_axis_direction("top")
    ax.axis["top"].label.set_text("Correlation")
    ax.axis["left"].set_axis_direction("bottom") 
    ax.axis["left"].label.set_text("Standard deviation")
    ax.axis["right"].set_axis_direction("top")   
    ax.axis["right"].toggle(ticklabels=True)
    ax.axis["right"].major_ticklabels.set_axis_direction("left")
    ax.axis["bottom"].set_visible(False)         
    ax.grid()
    polar_ax = ax.get_aux_axes(trans)   
    t = np.linspace(0,np.pi/2)
    r = np.zeros_like(t) + 1
    polar_ax.plot(t,r,'k--')
    polar_ax.text(np.pi/2+0.042,1.03, " 1.00", size=10.5,ha="right", va="top",
                  bbox=dict(boxstyle="square",ec='w',fc='w'))
    return polar_ax

def plot_taylor(axes, refsample, sample, *args, **kwargs):
    std = np.std(sample)
    corr = np.corrcoef(refsample, sample) 
    theta = np.arccos(corr[0,1])
    t,r = theta,std
    d = axes.plot(t,r, *args, **kwargs) 
    return d


import numpy as np
from plotTaylor import set_tayloraxes, plot_taylor
import matplotlib.pyplot as plt

x = np.linspace(0,10*np.pi,100)
data = np.sin(x)                           
m1 = data + 0.4*np.random.randn(len(x))    
m2 = 0.3*data + 0.6*np.random.randn(len(x)) 
m3 = np.sin(x-np.pi/10)                    
fig = plt.figure(figsize=(10,4))
ax1 = set_tayloraxes(fig, 121)
ax2 = set_tayloraxes(fig, 122)
print(data)
print(m1)
d1 = plot_taylor(ax2,data,m1, 'bo')
d2 = plot_taylor(ax2,data,m2, 'ro')
d3 = plot_taylor(ax2,data,m3, 'go')
plt.show()