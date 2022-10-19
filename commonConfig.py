from re import S
import numpy as np
import pandas as pd
import os
from pathlib import Path
from cdo import *
# from .script import TraceMeLand
from script.TraceMeLand import TraceMeLand
from netCDF4 import Dataset
# from multiprocessing import Pool 

cdo = Cdo()
pd.set_option('max_colwidth',200)

scripts = {
    'TraceMeLand': TraceMeLand
}

n_pool = os.cpu_count() - 1

class objCase:
    ## define the property
    def __init__(self,**keywords):
        # initialize
        self.dataSource      = keywords.get("dataSource",None)
        self.pathTemp        = keywords.get("pathTemp",None)
        # self.modelNames      = None # dictConf['datasets']['model']
        self.timeBnd         = keywords.get("timeBnd",None)
        self.nlatlon         = keywords.get("nlatlon",None)
        self.script          = keywords.get("script",None)
        self.areaFile        = 'area.nc'
        self.vegType         = 'VegetationType_1degree.nc'
        self.workDir         = keywords.get("workDir",None)
        self.traceProes      = keywords.get("traceProes",None)
        # ----------------------------------------------------
        self.startTime       = self.timeBnd[0]
        self.endTime         = self.timeBnd[1]
        self.latmin          = self.nlatlon[0]
        self.latmax          = self.nlatlon[1]
        self.lonmin          = self.nlatlon[2]
        self.lonmax          = self.nlatlon[3]
  
    
    def month2year(self):
        print('# change monthly to yearly ...')
        # if self.dataSource['frequency'].unique() == 'month':
        if self.pathTemp is not None:
            data2yearly = self.pathTemp + '/cdo_0_2yearly/'
        if not Path(data2yearly).exists(): os.makedirs(data2yearly)
        for index, idata in self.dataSource.iterrows():
            if idata.frequency != "year":
                if idata.path is not np.nan:
                    outName   = data2yearly  + idata.path.split('/')[-1].split('.nc')[0] + "_2yearly.nc"
                    cdo.yearmonmean(input = idata.path, output = outName)
                    self.dataSource.loc[index,'path'] = outName
                self.dataSource.loc[index,'frequency'] = "year"

    def timeExtraction(self):#, i_file, outpath, timestep):
        print("# time extraction ...")
        # for i_model in self.modelNames:
        if self.pathTemp is not None:
            timeExOutFile = self.pathTemp + '/cdo_1_timeExtraction/'
        if not Path(timeExOutFile).exists(): os.makedirs(timeExOutFile)
        for index, idata in self.dataSource.iterrows():
            if idata.path is not np.nan:
                outName   = timeExOutFile  + idata.path.split('/')[-1].split('.nc')[0] + "_SelTime.nc"
                stepStart = self.startTime - idata.start_year + 1 #### whether or not +1??
                stepEnd   = self.endTime   - idata.start_year + 1 
                if self.dataSource['frequency'].unique() == 'month':
                    stepStart = (stepStart - 1) * 12 + 1
                    stepEnd   = stepEnd * 12 
                cdo.seltimestep(str(int(stepStart))+'/'+str(int(stepEnd)), input = idata.path, output = outName)
                self.dataSource.loc[index,'path'] = outName 

    def regrid2onedegree(self):
        print("# change to one degree ...")
        if self.pathTemp is not None:
            oneGridOutFile = self.pathTemp+'/cdo_2_regrid2onedegree/'
        if not Path(oneGridOutFile).exists(): os.makedirs(oneGridOutFile)
        for index, idata in self.dataSource.iterrows():
            if idata.path is not np.nan:
                outName = oneGridOutFile + idata.path.split('/')[-1].split('.nc')[0] + "_r2oneDegree.nc"
                cdo.remapbil('r360x180', input=idata.path, output= outName)
                self.dataSource.loc[index, 'path'] = outName

    def spaceExtraction(self):
        print("# space extraction ...")
        if self.pathTemp is not None:
            spaceExOutFile = self.pathTemp + '/cdo_3_spaceExtraction/'
        if not Path(spaceExOutFile).exists(): os.makedirs(spaceExOutFile)
        for index, idata in self.dataSource.iterrows():
            if idata.path is not np.nan:
                outName = spaceExOutFile + idata.path.split('/')[-1].split('.nc')[0] + "_SelSpace.nc"
                cdo.sellonlatbox(self.lonmin,self.lonmax-1,self.latmin,self.latmax, input = idata.path, output = outName)
                self.dataSource.loc[index,'path'] = outName
        cdo.sellonlatbox(self.lonmin,self.lonmax,self.latmin,self.latmax, input = 'area.nc',    output = self.pathTemp+'/area.nc')
        cdo.sellonlatbox(self.lonmin,self.lonmax,self.latmin,self.latmax, input = self.vegType, output = self.pathTemp+'/vegetationType.nc')
        self.areaFile = self.pathTemp + '/area.nc'
        self.vegType  = self.pathTemp + '/vegetationType.nc'
    
    # def unitChange(inUnit, outUnit):
    #     if inUnit == "kg m-2":

    def runScript(self):
        print("# Run DIY script, such as Traceability analysis.")
        print(self.script)
        print(self.script in scripts)
        if self.script in scripts:
            scriptName = scripts[self.script]
        else:
            print("# Your script is not listed!")
            exit(0)
        region   = [self.latmin, self.latmax, self.lonmin, self.lonmax]
        timeStep = [self.startTime, self.endTime]
        # TraceMeLand(dataSource,region,timeBnd,areaFile,vegType,workDir,traceSpt="y",traceTem="y",traceVt='y',traceAB="y")
        # traceSpt="y",traceTem="y",traceVt='y',traceAB="y"
        scriptName(dataSource = self.dataSource, region  = region,       timeBnd = timeStep, 
                   areaFile   = self.areaFile,   vegType = self.vegType, workDir = self.workDir,
                   traceProes = self.traceProes)