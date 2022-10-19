
import numpy as np
from netCDF4 import Dataset
import netCDF4 as nc
from scipy.optimize import minimize
import multiprocessing as mp
from pathlib import Path
import os #, shutil
from .TraceMePlots import TraceMePlots
import time

import warnings

warnings.simplefilter("ignore")
olderr = np.seterr(all='ignore')

# preset the positional of components: can not be changed !!!!
x,   xc,  xp,  npp, tau, gpp, cue, bstau, senv, stas, spr, tas, pr = range(13)
abx, abxc, abxp, abtau, abstau = range(5)
#============================================================================#
def readNcFile(filepath, variablename): 
    nc_obj = Dataset(filepath)
    data   = np.array((nc_obj.variables[variablename][:]).data,dtype=float)
    try:
        data[data == nc_obj.variables[variablename].missing_value] = np.nan
    except AttributeError as e:
        print("Read nc-files: ", e)
    return data

def readVarData(dataSource, iModel, iVar):
    datLoc = dataSource.loc[(dataSource['model']==iModel)&(dataSource['variable'] == iVar)] # file locate
    if datLoc['path'].values[0] is np.nan:
        print("The variable " + iVar + " of " + iModel + " is nan !")
        results = 0
    else:
        results = readNcFile(datLoc['path'].values[0], iVar)
        if datLoc['unit'].values[0] == "kg m-2 s-1":          results *= 365*24*60*60 # maybe put it to preprocessing
        if iVar == "tas" and datLoc['unit'].values[0] == "K": results -= 273.15
    return results

def writeNcFile(ls_varName,dat4ncFile,outName,dimLen,dimName="spatial"):
    #save nc dat4ncFile[variable,data]
    #===create files and save
    # n_model = 0
    # for modelName in ls_modelName:
    da = nc.Dataset(outName,"w",format="NETCDF4") # create nc-files
    if dimName =="temporal":
        da.createDimension("time",dimLen[0,1]-dimLen[0,0]) # create Dimension: temporal(time); region(time,nlat,nlon)
        da.createVariable('time',"f8",("time"))
        da.variables["time"][:]= range(int(dimLen[0,0]),int(dimLen[0,1]))
        # create variables
        for i, varName in enumerate(ls_varName):
            da.createVariable(varName,"f8",("time"))
            da.variables[varName][:] = dat4ncFile[i,:] # dat4ncFile[modelName,varName,data]
    
    elif dimName == "spatial":
        da.createDimension("time",    dimLen[0,1]-dimLen[0,0]) # create Dimension: temporal(time); region(time,nlat,nlon)
        da.createDimension("latsize", dimLen[1,1]-dimLen[1,0])
        da.createDimension("lonsize", dimLen[2,1]-dimLen[2,0])
        da.createVariable("lat", "f8", ("latsize"))
        da.createVariable("lon", "f8", ("lonsize"))
        da.createVariable("time", "f8", ("time"))
        #da.createVariable("levels", "f8", ("levels"))
        da.variables["time"][:] = range(int(dimLen[0,0]),int(dimLen[0,1])) # range(start_year,end_year)
        da.variables["lat"][:]  = range(int(dimLen[1,0]),int(dimLen[1,1])) # range(latmin, latmax)
        da.variables["lon"][:]  = range(int(dimLen[2,0]),int(dimLen[2,1])) # range(lonmin, lonmax)
        for i, varName in enumerate(ls_varName):
            da.createVariable(varName, "f8", ("latsize", "lonsize"))
            da.variables[varName][:] = dat4ncFile[i,:] #convertLon(dat4ncFile[i,:],np.array([int(dimLen[1,0]),int(dimLen[1,1]),int(dimLen[2,0]),int(dimLen[2,1])]))
    else:
        return "WriteNcFile is filed. Error: dimName is "+dimName+" , which must be 'spatial' or 'temporal'."
    da.createdate= time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    da.close()
    return "write nc-file is finished!"

# ================ save nc-files =======================
# def func_saveNcFile():

############### calculate traceable components ######################
def calSptBsTau(datTas, datPr, datTau, numCores ):
    """
        run minimize based on each grid
        return: baseline residence time and Q10
    """
    global tmpTas, tmpPr, tmpTau, lsGrid
    tmpTas, tmpPr, tmpTau = datTas, datPr, datTau
    nlat, nlon = datTau.shape[1], datTau.shape[2]
    lsGrid = []
    for ilat in range(nlat):
        for ilon in range(nlon):
            lsGrid.append((ilat,ilon))
    if numCores > 3:
        pool    = mp.Pool(numCores - 1)
        try:
            results = pool.map(eachGrid4SptBsTau,[iGrid for iGrid in lsGrid]) 
            pool.close()
            pool.join()
        except Exception as e:
            print(e)
            print("Parallel run is error ....")
            results = []
            for iGrid in lsGrid:
                results.append(eachGrid4SptBsTau(iGrid))
    else:
        results = []
        for iGrid in len(lsGrid):
            results.append(eachGrid4SptBsTau(iGrid))
    del tmpTas, tmpPr, tmpTau
    reBsTau = np.full((nlat,nlon), np.nan)
    reQ10   = np.full((nlat,nlon), np.nan)
    for idxGrid, iGrid in enumerate(lsGrid):
        reBsTau[iGrid[0], iGrid[1]] = results[idxGrid][0]
        reQ10[iGrid[0], iGrid[1]]   = results[idxGrid][1]
    datTasMax = np.max(datTas, axis=0)
    datSTas   = np.power(reQ10,((datTas-datTasMax)/10))
    datSPr    = datPr/np.max(datPr, axis=0)
    datSEnv   = np.mean(np.array(datSTas)*np.array(datSPr), axis=0) 
    return reBsTau,datSEnv,np.mean(datSTas, axis=0),np.mean(datSPr, axis=0)

def eachGrid4SptBsTau(iGrid):
    datTas = tmpTas[:,iGrid[0],iGrid[1]]
    datPr  = tmpPr[:,iGrid[0],iGrid[1]]
    datTau = tmpTau[:,iGrid[0],iGrid[1]]
    results = calBaselineTau(datTas, datPr, datTau) # baseline tau; Q10
    return results

def func_cost(x,v_tem,max_tem,v_pre,max_pre,num,v_resTime):
    Q10        = x[0]
    v_based    = x[1]
    s_tem      = np.power(Q10,((v_tem-max_tem)/10))
    s_pre      = v_pre/max_pre
    total_s    = np.array(s_tem)*np.array(s_pre)
    r2         = 1-(sum(np.power((v_resTime - v_based/total_s),2))/(sum(np.power(v_resTime-v_based,2)))) #
    v_rmse     = np.linalg.norm(sum(np.power(v_resTime-v_based/total_s,2))/num)
    func       = abs(v_rmse/r2)
    return func

def calBaselineTau(datTas, datPr, datTau):
    numTime   = len(datTas)
    datTasMax = np.max(datTas)
    datPrMax  = np.max(datPr)
    x0        = np.array((1.0, np.min(datTau))) # 
    cons      = ({'type': 'ineq', 'fun': lambda x: x[0] -0.01 },
                 {'type': 'ineq', 'fun': lambda x: 10  - x[0]},
                 {'type': 'ineq', 'fun': lambda x: x[1]},
                 {'type': 'ineq', 'fun': lambda x: np.max(datTau)-x[1]})
    res       = minimize(lambda x:func_cost(x, datTas, datTasMax, datPr, datPrMax, numTime, datTau), x0,  constraints=cons)
    if res.success:
        Q10       = res.x[0]
        bTau      = res.x[1]
    else:
        Q10       = 2
        s_tem     = np.power(Q10,((datTas - datTasMax)/10))
        s_pre     = datPr/datPrMax
        total_s   = np.array(s_tem)*np.array(s_pre)
        bTau      = np.mean(datTau*total_s)
    return bTau, Q10

def calPartComponents(datX, datRateX, datNPP, datGPP):
    datTau = np.where(datNPP - datRateX == 0, np.nan, datX/(datNPP-datRateX))
    datXc  = datTau * datNPP
    datXp  = datXc - datX
    return datXc, datXp, datTau

def calComponents(datX, datRateX, datNPP, datGPP, datTas, datPr, temOrSpt="spt"):
    """
        return datXc, datXp, datTau, datCUE, datBsTau, datSEnv, datSTas, datSPr
        temOrSpt: spt/tem. run spatial/temporal baseline Tau
    """
    datXc, datXp, datTau = calPartComponents(datX, datRateX, datNPP, datGPP)
    datCUE = np.where(datGPP == 0, np.nan, datNPP/datGPP)
    if temOrSpt.lower() == "spt":
        datBsTau, datSEnv, datSTas, datSPr = calSptBsTau(datTas, datPr, datTau, int(mp.cpu_count()))
    elif temOrSpt.lower() == "tem":
        BsTau, Q10 = calBaselineTau(datTas, datPr, datTau)
        datBsTau   = np.repeat(BsTau, len(datTau))
        datSTas    = np.power(Q10,((datTas-np.max(datTas))/10))
        datSPr     = datPr/np.max(datPr)
        datSEnv    = np.array(datSTas)*np.array(datSPr)
    return datXc, datXp, datTau, datCUE, datBsTau, datSEnv, datSTas, datSPr

################# different traceable object: spt, tem, vegType ######################
#============================= functions =============================================
def func_averageMean(data, datArea): # nlat, nlon
    maskData = data * datArea
    datMa    = np.ma.MaskedArray(data,    mask = np.isnan(maskData)|np.isinf(maskData))
    datMaW   = np.ma.MaskedArray(datArea, mask = np.isnan(maskData)|np.isinf(maskData))
    reData   = np.ma.average(datMa, weights = datMaW) 
    return reData

def func_CalVegTypeComponents(data, lsTypeNum, datVegTypeEx, datArea, way = "sum", test="n"):
    nVeg   = len(lsTypeNum)
    nTime  = data.shape[0]
    reData = np.full((nVeg,nTime), np.nan)
    for index, iType in enumerate(lsTypeNum):
        if len(iType) == 1:
            dataTemp = np.where(datVegTypeEx == iType[0], data, np.nan) # time, nlat, nlon
        elif len(iType) >1:
            dataTemp = np.full((len(iType),data.shape[0],data.shape[1],data.shape[2]), np.nan)
            for idxNum, iNum in enumerate(iType):
                dataTemp[idxNum, :] = np.where(datVegTypeEx == iNum, data, np.nan) # time, nlat, nlon
            dataTemp = np.nansum(dataTemp,axis=0)
        dataTemp_tem = np.full((nTime), np.nan)
        if way == "sum":
            dataTemp_tem = np.nansum(np.nansum(dataTemp*datArea, axis = 1), axis=1) # timely
        else:
            for iTime in range(nTime):
                dataTemp_tem[iTime] = func_averageMean(dataTemp[iTime,:], datArea)
        reData[index,:] = dataTemp_tem
    return reData

#================= calculate: spt, tem, vegType, aboveground, belowground =================================
def calSptComponents(datX, datNPP, datGPP, datTas, datPr): # time, nlat, nlon
    reData           = np.full((13,datX.shape[1], datX.shape[2]), np.nan)
    datRateX         = datX[1:,:] - datX[:-1,:]
    results          = calComponents(datX[1:,:],datRateX,datNPP[1:,:],datGPP[1:,:],datTas[1:,:],datPr[1:,:])
    reData[x, :]     = np.mean(datX,         axis=0) # reData[0, :] = np.mean(datX,         axis=0) # x       
    reData[xc, :]    = np.mean(results[0],   axis=0) # reData[1, :] = np.mean(results[0],   axis=0) # xc
    reData[xp, :]    = np.mean(results[1],   axis=0) # reData[2, :] = np.mean(results[1],   axis=0) # xp
    reData[npp, :]   = np.mean(datNPP,       axis=0) # reData[3, :] = np.mean(datNPP,       axis=0) # npp
    reData[tau, :]   = np.mean(results[2],   axis=0) # reData[4, :] = np.mean(results[2],   axis=0) # tau
    reData[gpp, :]   = np.mean(datGPP[1:,:], axis=0) # reData[5, :] = np.mean(datGPP[1:,:], axis=0) # gpp
    reData[cue, :]   = np.mean(results[3],   axis=0) # reData[6, :] = np.mean(results[3],   axis=0) # cue
    reData[bstau, :] = results[4]                    # reData[7, :] = results[4]                    # bstau
    reData[senv, :]  = results[5]                    # reData[8, :] = results[5]                    # sEnv
    reData[stas, :]  = results[6]                    # reData[9, :] = results[6]                    # sTas
    reData[spr,:]    = results[7]                    # reData[10,:] = results[7]                    # sPr
    reData[tas,:]    = np.mean(datTas,       axis=0) # reData[11,:] = np.mean(datTas,       axis=0) # Tas
    reData[pr,:]     = np.mean(datPr,        axis=0) # reData[12,:] = np.mean(datPr,        axis=0)
    return reData # variable, nlat, nlon

def calTemComponents(datX, datNPP, datGPP, datTas, datPr, datArea):
    reData        = np.full((13,datX.shape[0]-1), np.nan)
    reData[npp,:] = np.nansum(np.nansum(datNPP[1:,:] * datArea, axis=1), axis=1)/1e12
    reData[gpp,:] = np.nansum(np.nansum(datGPP[1:,:] * datArea, axis=1), axis=1)/1e12
    tmpX          = np.nansum(np.nansum(datX         * datArea, axis=1), axis=1)/1e12 # transfer kg/m2 to PgC 
    for i in range(datX.shape[0]-1):
        reData[tas,i] = func_averageMean(datTas[i+1,:], datArea) 
        reData[pr,i]  = func_averageMean(datPr[i+1,:],  datArea)
    # lsIdx              = [xc, xp, tas, bst, sen, sta, spr] # dtXc, dtXp, dtTau, dtCUE, dtBsTau,dtSEnv, dtSTas, dtSPr
    results       = calComponents(tmpX[1:], tmpX[1:]-tmpX[:-1], reData[npp,:], reData[gpp,:], reData[tas,:], reData[pr,:], temOrSpt="tem")
    reData[x,:]   = tmpX[1:]
    reData[xc, :] = results[0]; reData[bstau,:] = results[4]
    reData[xp, :] = results[1]; reData[senv,:]  = results[5]
    reData[tau,:] = results[2]; reData[stas,:]  = results[6]
    reData[cue,:] = results[3]; reData[spr,:]   = results[7]
    return reData

def calVegTypeComponents(datX, datNPP, datGPP, datTas, datPr, lsTypeNum, datVegTypeEx, datArea):
    nVeg         = len(lsTypeNum)
    nTime        = datX.shape[0]
    reData         = np.full((13, nVeg, nTime-1), np.nan)
    tmpX           = func_CalVegTypeComponents(datX, lsTypeNum, datVegTypeEx,datArea)/1e12                     # [iVeg, time]
    reData[npp,:]  = (func_CalVegTypeComponents(datNPP, lsTypeNum, datVegTypeEx,datArea)/1e12)[:,1:]            # [iVeg, time]
    reData[gpp,:]  = (func_CalVegTypeComponents(datGPP, lsTypeNum, datVegTypeEx,datArea)/1e12)[:,1:]            # [iVeg, time]
    reData[tas,:]  = (func_CalVegTypeComponents(datTas, lsTypeNum, datVegTypeEx,datArea, way="average", test = "y"))[:,1:]  # [iVeg, time]
    reData[pr,:]   = (func_CalVegTypeComponents(datPr, lsTypeNum, datVegTypeEx,datArea,  way="average"))[:,1:]  # [iVeg, time]
    for iVeg in range(nVeg):
        results = calComponents(tmpX[iVeg,1:], tmpX[iVeg,1:]-tmpX[iVeg,:-1],reData[npp,iVeg,:], 
                                reData[gpp,iVeg,:],reData[tas,iVeg,:],reData[pr,iVeg,:], temOrSpt="tem") # xc, xp, tau, cue, bsTau, senv, stas, spr
        reData[xc,iVeg,:]   = results[0]; reData[bstau,iVeg,:] = results[4]
        reData[xp,iVeg,:]   = results[1]; reData[senv,iVeg,:]  = results[5]
        reData[tau,iVeg,:]  = results[2]; reData[stas,iVeg,:]  = results[6]
        reData[cue,iVeg,:]  = results[3]; reData[spr,iVeg,:]   = results[7]
    reData[x,:]    = tmpX[:,1:]
    return reData

def calSptABGroundComponents(datX, datNPP, datGPP, datLandX, datLandBsTau): #return abx, abxc, abxp, abtau, abbstau
    # input: above/below X, rdNPP, rdGPP, rd
    reData   = np.full((5, datX.shape[1], datX.shape[2]), np.nan)
    datRateX = datX[1:,:] - datX[:-1,:]
    results  = calPartComponents(datX[1:,:], datRateX, datNPP, datGPP)
    reData[abx,:]    = np.mean(datX[1:,:],axis=0)
    reData[abxc, :]  = np.mean(results[0],axis=0)
    reData[abxp, :]  = np.mean(results[1], axis=0)
    reData[abtau,:]  = np.mean(results[2], axis=0)
    reData[abstau,:] = (reData[abx,:]/datLandX)*datLandBsTau
    return reData

def calTemABGroundComponents(datX, datNPP, datGPP, datLandX, datLandBsTau):
    nTime    = len(datX)
    reData   = np.full((5, nTime-1),np.nan)
    datRateX = datX[1:]-datX[:-1]
    results  = calPartComponents(datX[1:], datRateX, datNPP, datGPP)
    reData[abx,   :] = datX[1:]
    reData[abxc,  :] = results[0]
    reData[abxp,  :] = results[1]
    reData[abtau, :] = results[2]
    reData[abstau,:] = (np.mean(datX[1:])/datLandX)*datLandBsTau
    return reData
    
def calVegTypeABGroundComponents(datX, datNPP, datGPP, datLandX, datLandBsTau):
    nVeg, nTime = datX.shape[0], datX.shape[1]
    reData      = np.full((5, nVeg, nTime-1), np.nan)
    datRateX    = datX[:,1:] - datX[:,:-1]
    results     = calPartComponents(datX[:,1:], datRateX,datNPP, datGPP)
    reData[abx,   :] = datX[:,1:]
    reData[abxc,  :] = results[0]
    reData[abxp,  :] = results[1]
    reData[abtau, :] = results[2]
    reData[abstau,:] = np.repeat((np.mean(datX[:,1:], axis=1)/np.mean(datLandX[:,1:], axis=1))[:,np.newaxis], nTime-1, axis=1)*datLandBsTau
    return reData

#================= Calculate variation decomposition =========================================================
# ----------------------------------------------------------------------------------------------------------------- #
class VarDecompObj():
    def __init__(self, inDataPath4R, resultPath4R, modelNames):
        self.RscriptSpt   = "Rscript "+ "\"" + "script/R_docs/RegionTAT_hier_part.R" + "\""
        self.RscriptTem   = "Rscript "+ "\"" + "script/R_docs/AnnualTAT_hier_part.R" + "\""
        self.inPath       = inDataPath4R
        self.outPath      = resultPath4R
        self.modelNames   = modelNames

    def run_temRscript(self, data, iThree = "Land"): 
        """ data: [components, model]"""
        temInPath  = self.inPath  + "data4Rscript_temporal/"
        temOutPath = self.outPath + "results_temporal/"
        self.func_checkDir([temInPath,temOutPath])
        inDataFile  = temInPath  + "AnnualTAT_data4vc_"+iThree+".csv"
        np.savetxt(inDataFile, data, delimiter = ',')
        outFileName = temOutPath + "variationDecomposition_temporal_"+iThree+".csv"
        self.func_OneRscriptTem(inDataFile,outFileName)
        return outFileName

    def run_temDynRscript(self, dtDynTem4R, iThree = "Land"): #
        """dtDynTem4R: [components, model, time]"""
        dynInPath  = self.inPath  + "data4Rscript_dynamic/" +iThree+ "/"
        dynOutPath = self.outPath + "results_dynamic/"      +iThree+ "/"
        self.func_checkDir([dynInPath,dynOutPath])
        self.parDynInDataPath  = dynInPath
        self.parDynOutFilePath = dynOutPath
        self.parDynInData      = dtDynTem4R
        nTime = dtDynTem4R.shape[2]
        numCores = int(mp.cpu_count())
        if numCores > 3:
            pool    = mp.Pool(numCores - 1)
            try:
                results = pool.map(self.func_parallelRunTem, [iTime for iTime in range(nTime)])
                pool.close()
                pool.join()
            except Exception as e:
                print("Parallel run is error ....")
                print(e)
                results = []
                for iTime in range(nTime):
                    results.append(self.func_parallelRunTem(iTime))
        else:
            results = []
            for iTime in range(nTime):
                results.append(self.func_parallelRunTem(iTime))
        return results
    
    def run_vegetationTypeVariationDecompTem(self, dtTemVt, lsVegTypes, iThree = "Land"):
        """ dtTemVt:  components, nVegType, nmodels """
        vtTemInPath  = self.inPath  + "data4Rscript_vegTypeTemporal/"
        vtTemOutPath = self.outPath + "results_vegTypeTemporal/"
        self.func_checkDir([vtTemInPath,vtTemOutPath])
        results = {}
        for idxVeg, iVeg in enumerate(lsVegTypes):
            # print("This is the data: ",dtTemVt[:, idxVeg,:])
            inDataFile  = vtTemInPath  + "AnnualTAT_data4vc_"+iVeg + "_" + iThree+".csv"
            np.savetxt(inDataFile, dtTemVt[:, idxVeg,:], delimiter = ',')
            outFileName = vtTemOutPath + "variationDecomposition_temporal_"+iVeg+"_"+iThree+".csv"
            results[iVeg] = self.func_OneRscriptTem(inDataFile,outFileName)
        return results

    def run_vegetationTypeVariationDecompDyn(self,dtDynVt, lsVegTypes, iThree = "Land"):
        """ dtDynVt:  components, nVegTypes, nModel, nTime"""
        results = {}
        for idxVeg, iVeg in enumerate(lsVegTypes):
            dynInPath  = self.inPath  + "data4Rscript_vegTypeDynamic/"+ iVeg + "_" +iThree+ "/"
            dynOutPath = self.outPath + "results_vegTypeDynamic/"     + iVeg + "_" +iThree+ "/"
            self.func_checkDir([dynInPath,dynOutPath])
            self.parDynInDataPath  = dynInPath
            self.parDynOutFilePath = dynOutPath
            self.parDynInData      = dtDynVt[:,idxVeg,:,:] 
            nTime = self.parDynInData.shape[2]
            numCores = int(mp.cpu_count())
            if numCores > 3:
                pool    = mp.Pool(numCores - 1)
                try:
                    results[iVeg] = pool.map(self.func_parallelRunTem, [iTime for iTime in range(nTime)])
                    pool.close()
                    pool.join()
                except Exception as e:
                    print("Parallel run is error ....")
                    print(e)
                    results[iVeg] = []
                    for iTime in range(nTime):
                        results[iVeg].append(self.func_parallelRunTem(iTime))
            else:
                results[iVeg] = []
                for iTime in range(nTime):
                    results[iVeg].append(self.func_parallelRunTem(iTime))
        return results

    def run_sptRscript(self, dtSpt4R, nlatlon, iThree="Land"):
        self.saveVarNames = ["carbon_storage_", "carbon_storage_capacity_", "carbon_storage_potential_", 
                            "npp_", "residence_time_", "gpp_", "cue_","baseline_residence_time_",
                            "temperature_", "rain_"]
        self.sptInPath  = self.inPath  + "data4Rscript_spatial/" +iThree+ "/"
        self.sptOutPath = self.outPath + "results_spatial/"      +iThree+ "/"
        self.func_checkDir([self.sptInPath,self.sptOutPath])
        self.datSpt4R   = dtSpt4R
        latmin, latmax, lonmin, lonmax = nlatlon
        # save data to csv-files for Rscript-spatial
        numCores = int(mp.cpu_count())
        if numCores > 3:
            pool        = mp.Pool(numCores - 2)
            pool.map(self.func_savetxt, [iVar for iVar in range(len(self.saveVarNames))]) # save each components
            pool.close()
            pool.join()
        else:
            for iVar in range(len(self.saveVarNames)):
                self.func_savetxt(iVar)
        outfilename = self.sptOutPath + "res_cv_spatial.csv"
        os.system(self.RscriptSpt+' '+ self.sptInPath +' '+str(len(self.modelNames))+' '+"\""+ outfilename + "\"" +' '+str(latmin)+' '+str(latmax)+' '+str(lonmin)+' '+str(lonmax))
        return outfilename

    def func_parallelRunTem(self, iTime):
        inDataFile = self.parDynInDataPath + "AnnualTAT_data4vc_"+str(iTime)+".csv"
        inData     = self.parDynInData[:,:,iTime]
        np.savetxt(inDataFile, inData, delimiter=',')
        outFileName = self.parDynOutFilePath + "resultCV_dynamic_"+ str(iTime) +".csv"
        self.func_OneRscriptTem(inDataFile, outFileName)
        return outFileName
    
    def func_savetxt(self, iVar):
        for iModel in range(len(self.modelNames)):
            fileSpt = self.sptInPath + self.saveVarNames[iVar] +str(iModel + 1) + ".csv"
            np.savetxt(fileSpt, self.datSpt4R[iVar,iModel,:], delimiter = ',')

    def func_OneRscriptTem(self,inDataFile,outFileName):
        if self.RscriptTem is not None and self.modelNames is not None:
            try:
                os.system(self.RscriptTem+" "+ "\""+inDataFile+"\"" + " "+str(len(self.modelNames))+" "+ "\""+outFileName+"\"")
            except Exception as e:
                print("run temporal variation decomposition: ",e)
        else:
            print("RscriptTem and/or modelNames is None. please check it.")
        return outFileName
    
    def func_checkDir(self, lsPath):
        for iPath in lsPath:
            iPath4Create = Path(iPath)
            if not iPath4Create.exists(): os.makedirs(iPath4Create)



#================= main traceability analysis submudule ======================================================
def TraceMeLand(dataSource,region,timeBnd,areaFile,vegType,workDir, traceProes): # dataSource is DataFrame
    print("traceProes:", traceProes)
    traceTem, traceSpt, traceVt, traceAB = traceProes   
    print("traceTem, traceSpt, traceVt, traceAB", traceTem, traceSpt, traceVt, traceAB)
    modelNames = dataSource['model'].unique()
    # zhoujian: test
    # modelNames = modelNames[:3]
    varNames   = ['cCwd', 'cVeg', 'cLitter', 'cSoil', 'npp', 'gpp', 'tas', 'pr']        # variable names for reading nc-files; readVariables = dataSource['variable'].unique()
    latmin, latmax, lonmin, lonmax = region
    nlat, nlon = latmax - latmin, lonmax-lonmin
    nTime      = timeBnd[1] - timeBnd[0] + 1 
    lsVegTypes = ["ENF","EBF","DNF","DBF","MF","Shrub","Sav","Grass","Tundra","Barren"]
    lsTypeNum  = [[1],   [2],  [3],  [4],  [5], [6,7], [8,9], [10],    [15],    [16]] 
    datArea            = readNcFile(areaFile,'area')*1e6
    datArea[datArea<0] = np.nan
    datVegType    = readNcFile(vegType,"VegetationType")
    datVegTypeEx  = np.repeat(datVegType[np.newaxis,:], nTime, axis=0)
    if traceSpt: trcDtSp = np.full((13,len(modelNames),nlat,nlon), np.nan)
    if traceTem: trcDtTm = np.full((13,len(modelNames),nTime-1), np.nan)
    if traceVt:  trcDtVt = np.full((13,len(lsTypeNum),len(modelNames),nTime-1), np.nan)
    if traceAB:
        trcSptAg = np.full((5,len(modelNames),nlat,nlon), np.nan)
        trcSptBg = np.full((5,len(modelNames),nlat,nlon), np.nan) 
        trcTemAg = np.full((5,len(modelNames),nTime-1), np.nan)
        trcTemBg = np.full((5,len(modelNames),nTime-1), np.nan)
        if traceVt:
            trcVtAg  = np.full((5,len(lsTypeNum),len(modelNames),nTime-1), np.nan) 
            trcVtBg  = np.full((5,len(lsTypeNum),len(modelNames),nTime-1), np.nan)
    for idxModel, iModel in enumerate(modelNames):
        rdCwd  = readVarData(dataSource, iModel, "cCwd")
        rdLit  = readVarData(dataSource, iModel, "cLitter")
        rdVeg  = readVarData(dataSource, iModel, "cVeg")
        rdSoil = readVarData(dataSource, iModel, "cSoil")
        rdNPP  = readVarData(dataSource, iModel, "npp")
        rdGPP  = readVarData(dataSource, iModel, "gpp")
        rdTas  = readVarData(dataSource, iModel, "tas")
        rdPr   = readVarData(dataSource, iModel, "pr")
        datSpX = rdCwd + rdLit + rdVeg + rdSoil
        if traceAB: datSpXAg = rdVeg; datSpXBg = rdCwd+rdLit + rdSoil
        del rdCwd, rdLit, rdVeg, rdSoil
        trcDtSp[:,idxModel,:,:] = calSptComponents(datSpX, rdNPP, rdGPP, rdTas, rdPr)
        trcDtTm[:,idxModel,:]   = calTemComponents(datSpX, rdNPP, rdGPP, rdTas, rdPr, datArea)
        if traceVt:
            trcDtVt[:,:,idxModel,:] = calVegTypeComponents(datSpX, rdNPP, rdGPP, rdTas,rdPr,lsTypeNum,datVegTypeEx,datArea)
        if traceAB:
            trcSptAg[:,idxModel,:,:] = calSptABGroundComponents(datSpXAg,rdNPP[1:,:],rdGPP[1:,:],trcDtSp[x,idxModel,:,:],trcDtSp[bstau,idxModel,:,:])
            trcSptBg[:,idxModel,:,:] = calSptABGroundComponents(datSpXBg,rdNPP[1:,:],rdGPP[1:,:],trcDtSp[x,idxModel,:,:],trcDtSp[bstau,idxModel,:,:])
            # --------------------------------------------------------------------------------------------
            tmpX        = np.nansum(np.nansum(datSpXAg * datArea, axis=1), axis=1)/1e12 
            trcTemAg[:,idxModel,:] = calTemABGroundComponents(tmpX,trcDtTm[npp,idxModel,:],trcDtTm[gpp,idxModel,:],
                                                   trcDtTm[x,idxModel,:],trcDtTm[bstau,idxModel,:])
            # --------------------------------------------------------------------------------------------
            tmpX        = np.nansum(np.nansum(datSpXBg * datArea, axis=1), axis=1)/1e12  
            trcTemBg[:,idxModel,:] = calTemABGroundComponents(tmpX,trcDtTm[npp,idxModel,:],trcDtTm[gpp,idxModel,:],
                                                   trcDtTm[x,idxModel,:],trcDtTm[bstau,idxModel,:])
            if traceVt:
                tmpX = func_CalVegTypeComponents(datSpXAg, lsTypeNum, datVegTypeEx, datArea)/1e12  
                trcVtAg[:,:,idxModel,:] = calVegTypeABGroundComponents(tmpX,trcDtVt[npp,:,idxModel,:], trcDtVt[gpp,:idxModel,:],
                                            trcDtVt[x,:,idxModel,:], trcDtVt[bstau,:,idxModel,:])
                tmpX = func_CalVegTypeComponents(datSpXBg, lsTypeNum, datVegTypeEx, datArea)/1e12  
                trcVtBg[:,:,idxModel,:] = calVegTypeABGroundComponents(tmpX,trcDtVt[npp,:,idxModel,:], trcDtVt[gpp,:idxModel,:],
                                            trcDtVt[x,:,idxModel,:], trcDtVt[bstau,:,idxModel,:])
    #========= end read model data ==============================================================#
    print("# the model data read is finished!")
    # ----------------- Start to run variation decomposition --------------------------
    outDataPath4R = workDir + '/R_docs/'
    resultPath4R  = workDir + "/results/csv-files/"
    vdObj         = VarDecompObj(inDataPath4R=outDataPath4R,resultPath4R=resultPath4R,modelNames=modelNames)
    if traceTem:
        print("# Start to run variation decomposition: temporal ...")
        dat4temR       = np.mean(trcDtTm, axis=2)      # mean of timely data
        lsTemVCPath    = vdObj.run_temRscript(dat4temR)
        lsTemVCDynPath = vdObj.run_temDynRscript(trcDtTm)
    if traceSpt:
        print("# Start to run variation decomposition: spatial ...")
        lsSptVCPath =  vdObj.run_sptRscript(trcDtSp, region)
        print("lsSptVCPath:", lsSptVCPath)
    if traceVt:
        print("# Start to run variation decompositon: different vegetation types ...")
        datVtTem      = np.mean(trcDtVt, axis=3)
        lsVtVCPath    = vdObj.run_vegetationTypeVariationDecompTem(datVtTem, lsVegTypes)
        lsVtVCDynPath = vdObj.run_vegetationTypeVariationDecompDyn(trcDtVt,  lsVegTypes)
    # --------------- Start to run plotting scripts and nc-files -----------------------------
    print("# Start to run ploting and saving nc-files....")
    ls_varName = ["cSto","cStoCap","cStoPot","NPP","cTau", "GPP", "CUE", "BaseCtau", "senv" ,"sTas", "sPr","tas","pr"]
    ls_varNameAG = ["cStoAG","cStoCapAG","cStoPotAG","cTauAG","BaseCtauAG"]
    ls_varNameBG = ["cStoBG","cStoCapBG","cStoPotBG","cTauBG","BaseCtauBG"]
    plotObj = TraceMePlots(workDir+"/results/figures/", modelNames, varNames, timeBnd)
    if traceTem:
        plotObj.runUncertaintySource(lsTemVCPath, trcDtTm)    # multi-pie figures, sankey figure
        plotObj.runTemporalEvaluation(trcDtTm)
        plotObj.runDynamicVariationContribution(lsTemVCDynPath,timeBnd)
        dimLen      = np.full((1,2), np.nan)
        dimLen[0,0] = timeBnd[0]; dimLen[0,1] = timeBnd[1]
        for idx, iModel in enumerate(modelNames):     
            if traceAB:
                writeData = np.vstack((trcDtTm[:,idx,:], trcTemAg[:,idx,:], trcTemBg[:,idx,:]))
                writeVars = ls_varName + ls_varNameAG + ls_varNameBG
            else:
                writeData = trcDtTm[:,idx,:]
                writeVars = ls_varName
            outName = workDir+"/results/nc-files/temporal_traceability_analysis_"+iModel+".nc"
            writeNcFile(writeVars,writeData,outName,dimLen,dimName="temporal")
    if traceSpt:
        try:
            plotObj.runSpatialVariationContribution(lsSptVCPath, region)
        except Exception as e:
            print("run to plot spatial variation contribution is error: ", e)
        try:
            plotObj.runSptLatComponents(trcDtSp, datArea, region)
        except Exception as e:
            print("run to plot spatial latitude components is error: ",e)
        try:
            plotObj.runSptComponentsEvaluation(region, trcDtSp)
        except Exception as e:
            print("run to plot spatial components is error: ",e)
        dimLen      = np.full((3,2), np.nan)
        dimLen[0,0] = timeBnd[0]; dimLen[0,1] = timeBnd[1]
        dimLen[1,0] = latmin; dimLen[1,1] = latmax
        dimLen[2,0] = lonmin; dimLen[2,1] = lonmax  
        for idx, iModel in enumerate(modelNames):
            if traceAB:
                writeData = np.vstack((trcDtSp[:,idx,:,:], trcSptAg[:,idx,:,:], trcSptBg[:,idx,:,:]))
                writeVars = ls_varName + ls_varNameAG + ls_varNameBG
            else:
                writeData = trcDtSp[:,idx,:,:]
                writeVars = ls_varName
            outName = workDir+"/results/nc-files/spatial_traceability_analysis_"+iModel+".nc"
            writeNcFile(writeVars,writeData,outName,dimLen,dimName="spatial")
    
    if traceVt:
        try:
            plotObj.runVegTypeVariationContribution(lsVtVCPath)
        except Exception as e:
            print("runVegTypeVariationContribution is error: ",e)
        try:
            plotObj.runVegTypeComponentsEvaluation(trcDtVt, lsVegTypes)
        except Exception as e:
            print("runVegTypeComponentsEvaluation is error: ",e)
            
        for key, iFile in lsVtVCDynPath.items():
            try:
                plotObj.runDynamicVariationContribution(iFile,timeBnd, iVeg=key)
            except Exception as e:
                print("runDynamicVariationContribution is error: ",e)
                print(key)
                continue
        dimLen      = np.full((1,2), np.nan)
        dimLen[0,0] = timeBnd[0]; dimLen[0,1] = timeBnd[1]
        for idxVeg, iVeg in enumerate(lsVegTypes):
            trcDtVtIveg = trcDtVt[:,idxVeg,:,:]
            for idx, iModel in enumerate(modelNames):
                if traceAB:
                    writeData = np.vstack((trcDtVt[:,idxVeg,idx,:], trcVtAg[:,idxVeg,idx,:], trcVtBg[:,idxVeg,idx,:]))
                    writeVars = ls_varName + ls_varNameAG + ls_varNameBG
                else:
                    writeData = trcDtVt[:,idxVeg,idx,:]
                    writeVars = ls_varName
                outName = workDir+"/results/nc-files/vegetationType_traceability_analysis_"+iModel+"_"+iVeg+".nc"
                writeNcFile(writeVars,writeData,outName,dimLen,dimName="temporal")        