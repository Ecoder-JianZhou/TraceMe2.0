'''
    1. Read yml file
    2. Preset objCase: conf; date; output;
        conf: dateSource (path)
            work dir (path)
        date: modelname; variablename;startTime; endTime; handleScript
        output: object for figuring ()
'''
import os
import argparse
import script
import sys
import yaml
import os.path
from pathlib import Path
from commonConfig import objCase
import pandas as pd
import shutil

pd.set_option('max_colwidth',200)
# import logging

def mkWorkDir(myfpath):
    print("This is for creating the work dir: ", myfpath)
    print("# Creating the dirs ...")
    if Path(myfpath).exists():
        try:
            shutil.rmtree(myfpath,ignore_errors=True)
        except OSError:
            pass
    if not Path(myfpath+"/results").exists():os.makedirs(myfpath+"/results")
    if not Path(myfpath+"/results/figures").exists():os.makedirs(myfpath+"/results/figures")
    if not Path(myfpath+"/results/nc-files").exists():os.makedirs(myfpath+"/results/nc-files")
    if not Path(myfpath+"/temp").exists(): 
        os.makedirs(myfpath+"/temp")
    else:
        shutil.rmtree(myfpath+"/temp",ignore_errors=True)

def readInput(file_dir):    
    modelData = pd.DataFrame()
    ipath = {}
    for root, dirs, files in os.walk(file_dir):
        if len(files):
            for i_file in files: # read file name
                if os.path.splitext(i_file)[1] == '.nc': # make sure it is nc-format file
                    model   = root.replace(file_dir,'').split('/')[1]
                    varName = i_file.split('_')[0] # cLitter_yr_bcc-csm1-1-m_r1i1p1_185001-210012.nc
                    ipath['model']    = model
                    ipath['variable'] = varName
                    ipath['path']     = root+'/'+i_file
                    modelData = modelData.append(ipath,ignore_index =True)   
    return modelData

### Start: get yml file ###        
parser = argparse.ArgumentParser()
parser.add_argument('yml', type=str, help="Please input your config.yml")
args   = parser.parse_args()
# print(args)
### Read the config.yml --> cfg: {}
if len(args.yml)>4:
    if args.yml[-4:] == '.yml':
        ymlFileName = "yml/"+args.yml
    else:
        ymlFileName = "yml/"+args.yml+'.yml'
else:
    ymlFileName = "yml/"+args.yml+'.yml'

print("# Reading config info and input data info ...")
try:
    with open(ymlFileName) as file:
        cfg = yaml.safe_load(file)
except FileNotFoundError as e:
    #logger.error(e)
    print(e)

## DataFrame.name:modelname; exp; ensemble; start_year; end_year; variable; 
preDataFrame = pd.DataFrame()
for i,iterm in enumerate(cfg['datasets']):
    for j, i_varname in enumerate(cfg['variables']):
        dd = iterm
        dd['variable']  = i_varname['name']
        dd['frequency'] = i_varname['frequency'] 
        dd['unit']      = i_varname['unit']
        # dd['path']     = ""
        preDataFrame = preDataFrame.append(dd,ignore_index =True)    

### Get the input info from the input dir
### return the datasets into pd.dataframe
dataSource = readInput(cfg['path']['inputDir'])  # return a dist {"modelname":{"variables": {}}}
dataSource = dataSource.merge(preDataFrame,on=['model','variable'],how='outer')
if dataSource[dataSource.isnull().T.any()].shape[0]>0:
    print("Your preset info is error: ")
    print(dataSource[dataSource.isnull().T.any()])

start_year = dataSource['start_year'].max()
end_year   = dataSource['end_year'].min()
if start_year < end_year:
    print(start_year,end_year)
    pass
else:
    print("your time for evaluation is mismatch:", "start_year:", start_year,"; end year:", end_year)

## create the work dir
mkWorkDir(cfg['path']['workRoot'])
pathTemp    = cfg['path']['workRoot']+"/temp"
pathFigures = cfg['path']['workRoot']+"/results/figures"
pathNcFiles = cfg['path']['workRoot']+"/results/nc-files"

## run analysis processes:
traceTem   = cfg['analysis']['temporalAnalysis']
traceSpt   = cfg['analysis']['spatialAnalysis']
traceVt    = cfg['analysis']['VegetationType']
traceAB    = cfg['analysis']['aboveAndBelowGround']
traceProes = [traceTem, traceSpt, traceVt, traceAB]

print("# Start run case ...")
## create new case to proprocess
nlatlon = [cfg['region']['latmin'], cfg['region']['latmax'], cfg['region']['lonmin'], cfg['region']['lonmax']]
timeBnd = [int(start_year), int(end_year)]
testObj = objCase(dataSource = dataSource, 
                  pathTemp   = pathTemp, 
                  timeBnd    = timeBnd,
                  nlatlon    = nlatlon,
                  script     = cfg['script'],
                  workDir    = cfg['path']['workRoot'],
                  traceProes = traceProes)
## Preprocessing ##
testObj.month2year()
testObj.timeExtraction()
testObj.regrid2onedegree()
testObj.spaceExtraction()   
testObj.runScript()         # including plotting 