##################################################################################################################
# This submodule is for plotting the fugures to show the results of traceability analysis.
#
# Include:
#    1. Figures for traceability analyis results:
#       1.1 plotMultiPie: show the variance contribution of different traceable components.
#       1.2 plotSankey:   show the flow of traceable processes based on the relationship among traceable components
#       1.3 plotHeatmap:  show the relative variance of different components.
#    2. Figures for temporal traceability analysis results:
#       2.1 plotTemFracVC: plot temporal fraction of variation contribution of different components.
#       2.2 plotLines:     plot multi-model time-series dynamics
#       2.3 plotBox:       show the statistic of variation contribution
#       2.4 plotScatter2D: show the relationship between traceable components.
#       2.5 plotScatter3D: show the relationship among Tau, baseline Tau and environments
#    3. Figures for global traceability analysis results:
#       3.1 plotGlobMap:
#       3.2 plotGlobLat:
#    4. Figures for different ecosystems:
#    5. Figures for Benchmark analysis:
#       * global results
#       * latitude results
#       * temporal results: Taylor prography
#       * different ecosystems   
##################################################################################################################

import numpy as np
import pandas as pd
import script.TraceMePlotLibs as trcPlt

# preset the positional of components: can not be changed !!!!
x,   xc,  xp,  npp, tau, gpp, cue, bstau, senv, stas, spr, tas, pr = range(13)
abx, abxc, abxp, abtau, abstau = range(5)

class TraceMePlots:
    def __init__(self, figPath, modelNames, varNames, timeBnd, **keywords):
        self.figPath    = figPath
        self.modelNames = modelNames
        self.varNames   = varNames
        self.timeBnd    = timeBnd
        # self.datSpt     = keywords.get("datSpt", None)
        # self.datTem     = keywords.get("datTem", None)
        # self.detEcoTem  = keywords.get("datTem", None)
        # -----------------------------------------------
        self.components  = {x:"Carbon storage", xc:  "Carbon storage capacity", xp:   "Carbon storage potential", npp:"NPP", 
                            tau: "Carbon residence time", gpp: "GPP", cue:"CUE",          bstau:"Baseline carbon residence time",
                            senv:"Environmental scalars", stas:"Temperature scalar",      spr:  "Precipitation scalar", 
                            tas:"Temperature",            pr:  "Precipitation"}
        self.units       = {x:   'kg C m-2', xc:    'kg C m-2', xp: 'kg C m-2', npp: 'kg C m-2 year-1', tau:   'year', gpp:  'kg C m-2 year-1', 
                            cue: '-',        bstau: 'year',     senv: '-',      tas: 'degree',          pr:    'mm'}
        self.colorVars   = {x:  '#B8860B', xc:  '#D2691E', xp:   '#C0C0C0', npp:'#008000', tau:'#FFA500',
                            gpp:'#32CD32', cue: '#98FB98', bstau:'#F08080', tas:'#FFFF00', pr: '#1E90FF'}
        self.colorModels = trcPlt.mkColors(len(self.modelNames))[1:] # without observation
 
    def runUncertaintySource(self, inFile, datTem, iThree ="land"):
        ########  muti-pie: variation decomposition ############ 
        data       = pd.read_csv(inFile,index_col=False).iloc[:,1].values
        titleName  = "Variation contribution of traceable components\n (" +iThree +":" + str(self.timeBnd[0]) + "-" + str(self.timeBnd[1]) + ")"
        components = [self.components[xp],  self.components[xc],    self.components[npp], self.components[tau], self.components[gpp],
                      self.components[cue], self.components[bstau], self.components[tas], self.components[pr]]
        ##             xp   , xc, npp, tau, gpp, cue, base, T, W 
        colors     = [self.colorVars[xp], self.colorVars[xc],   self.colorVars[npp], self.colorVars[tau], self.colorVars[gpp], 
                      self.colorVars[cue],self.colorVars[bstau],self.colorVars[tas], self.colorVars[pr]]
        outFig     = self.figPath+ "figure_0-1_multiPie_"+iThree+".png" 
        trcPlt.plotMultPie(np.array(data),labels=components,colors=colors,titleName=titleName,outFig=outFig)
        ########## sankey plot ###############################################################################
        nodes = [ {"name": "C storage",           'itemStyle':{'color':'#B8860B'}},
                  {"name": "C storage capacity",  'itemStyle':{'color':'#D2691E'}},
                  {"name": "C storage potential", 'itemStyle':{'color':'#C0C0C0'}},
                  {"name": "NPP",                 'itemStyle':{'color':'#008000'}},
                  {"name": "Tau",                 'itemStyle':{'color':'#FFA500'}},
                  {"name": "CUE",                 'itemStyle':{'color':'#98FB98'}},
                  {"name": "GPP",                 'itemStyle':{'color':'#32CD32'}},
                  {"name": "Baseline Tau",        'itemStyle':{'color':'#F08080'}},
                  {"name": "Temperature",         'itemStyle':{'color':'#FFFF00'}},
                  {"name": "Precipitation",       'itemStyle':{'color':'#1E90FF'}},]
        colors = ['#B8860B','#D2691E', '#C0C0C0', '#008000','#FFA500',
                  '#98FB98','#32CD32', '#F08080', '#FFFF00','#1E90FF']
        links  = [{"source": "C storage",          "target": "C storage potential", "value": data[0]},
                  {"source": "C storage",          "target": "C storage capacity",  "value": data[1]}, 
                  {"source": "C storage capacity", "target": "NPP",                 "value": data[2]},
                  {"source": "C storage capacity", "target": "Tau",                 "value": data[3]},
                  {"source": "NPP",                "target": "CUE",                 "value": data[4]},
                  {"source": "NPP",                "target": "GPP",                 "value": data[5]},
                  {"source": "Tau",                "target": "Baseline Tau",        "value": data[6]},
                  {"source": "Tau",                "target": "Temperature",         "value": data[7]},
                  {"source": "Tau",                "target": "Precipitation",       "value": data[8]},]
        outFigHtml = self.figPath+"figure_0-2_sankey_"+iThree+".html"
        trcPlt.plotSankey(nodes, links, colors, "Traceability results", outFigHtml)
        ####  heatmap figures ######
        VarNames = ["C storage",    "C storage capacity","C storage potential", "NPP", "Tau", "CUE", "GPP",
                    "Baseline Tau", "Environments",      "Temperature",         "Precipitation"]
        datHeatMap = np.mean(datTem, axis=2) # datTem: variables, models, time
        for i in range(datHeatMap.shape[0]):
            datHeatMap[i,:] = (datHeatMap[i,:] - np.min(datHeatMap[i,:]))/(np.max(datHeatMap[i,:])-np.min(np.min(datHeatMap[i,:])))
            datHeatMap[i,:] = (datHeatMap[i,:] - np.mean(datHeatMap[i,:]))/(np.mean(datHeatMap[i,:]))
        lsIndex   = [0,1,2,3,4,5,6,7,8,11,12]
        data4Plot = datHeatMap[lsIndex,:]
        pdData = pd.DataFrame(data4Plot,columns=self.modelNames,index=VarNames)
        outFig = self.figPath + "figure_0-3_heatmap_"+iThree+".png"
        trcPlt.plotHeatmap(pdData,outFig,"Multi-Model Bias",[-1,1])

    def runTemporalEvaluation(self, datTem, iThree = "land"): # datTem: vars, model, time
        colors = self.colorModels
        #### temporal change of each components ####
        # -------- figure-1-0-1: npp ------------------
        lsPltVars = [npp, tau, gpp, cue, bstau, tas, pr]
        for iVar in lsPltVars:
            xylabel = ["Year", self.components[iVar]+" ("+self.units[iVar]+")"]
            outFig  = self.figPath + "figure-1-0_temporal_dynamic_of_" + self.components[iVar] + ".png"
            trcPlt.plotLines(range_x=self.timeBnd, dat_y=datTem[iVar,:,:], ls_modelName=self.modelNames,
                             colors=colors, xylabel=xylabel, outFig=outFig)
        # -------- figure-1-4: xc, xp and x -----------
        xylabel = ["Year","Carbon storage and capacity \n(Pg C)"]
        outFig  = self.figPath + "figure_1-4_xc_xp_x_" + iThree + ".png"
        trcPlt.plotLines(range_x=self.timeBnd, dat_y=datTem[x,:,:], ls_modelName=self.modelNames,
                         colors=colors, xylabel=xylabel, outFig=outFig, dat_yy=datTem[xc,:,:])
        # -------- figure-1-5: npp-tau ----------------
        xylabel = ["NPP (Pg C/year)","Residence time \n(year)"]
        outFig  = self.figPath + "figure_1-5_npp_tau_xc_" + iThree + ".png"
        trcPlt.plotScatter2D(dat_x=datTem[npp,:,:],dat_y=datTem[tau,:,:], ls_modelName=self.modelNames, colors=colors, xylabel=xylabel, outFig=outFig, Zway="x*y")
        # -------- figure-1-6: gpp-npp-cue ------------
        xylabel = ["NPP (Pg  C/year)","GPP (Pg C/year)"]
        outFig  = self.figPath + "figure_1-6_gpp_npp_cue_" + iThree + ".png"
        trcPlt.plotScatter2D(dat_x=datTem[npp,:,:], dat_y=datTem[gpp,:,:], ls_modelName=self.modelNames, colors=colors, xylabel=xylabel, outFig=outFig, Zway="x/yD")
        # -------- figure-1-7: tau-baselineTau-SEnv ---
        xylabel = ["Baseline residence time (year)", "Environmental scalars"]
        outFig  = self.figPath + "figure_1-7_bas_env_tau" + iThree + ".png"
        trcPlt.plotScatter2D(dat_x=datTem[bstau,:,:], dat_y=datTem[senv,:,:], ls_modelName=self.modelNames, colors=colors, xylabel=xylabel, outFig=outFig, Zway="x/y")
        # -------- figure-1-8: 
        xylabel      = ["Temperature (degree)","Precipitation (mm)"]
        outFig  = self.figPath + "figure_1-8_tem_pr_" + iThree + ".png"
        trcPlt.plotScatter2D(dat_x=datTem[tas,:,:], dat_y=datTem[pr,:,:], ls_modelName=self.modelNames, colors=colors, xylabel=xylabel, outFig=outFig)
    
    def runDynamicVariationContribution(self, lsInFiles, timeBnd, iVeg = None, iThree="land"): # each year cv.csv
        ####  dynamic of variation contribution ###################### 
        lineNames = ["Carbon storage potential", "GPP", "CUE", "Baseline residence time", "Temperature", "Precipitation"]
        colors    = ['#C0C0C0', '#32CD32', '#98FB98', '#F08080', '#FFFF00', '#1E90FF']
        xyLabels  = ["Time", "Fraction of total variance (%)"]
        data3vars = np.zeros((3,len(lsInFiles)))
        data      = np.zeros((6,len(lsInFiles)))
        for i, iFile in enumerate(lsInFiles):
            tmp = pd.read_csv(iFile,index_col=False).iloc[:,1].values # xp, xc, npp/xc, tau/xc, gpp, cue, bstau, tas, pr
            data3vars[0,i] = tmp[0]; data3vars[1,i] = tmp[2]; data3vars[2,i] = tmp[3]
            data[0,i] = tmp[0]; data[1,i] = tmp[4]; data[2,i] = tmp[5]
            data[3,i] = tmp[6]; data[4,i] = tmp[7]; data[5,i] = tmp[8]
        x    = range(timeBnd[0], timeBnd[1])
        #--- figure-1: the variation contribution dynamic of each components ------------------------------------------------------
        if iVeg is None:
            outPath = self.figPath + "figure_1-1_dynamic_variation_contribution_"+iThree+".png"
        else:
            outPath = self.figPath + "figure_1-1_dynamic_variation_contribution_"+iVeg+"_"+iThree+".png"
        line3names = ["Carbon storage potential", "NPP", "Carbon residence time"]
        colors3var = ['#C0C0C0', '#008000','#FFA500']
        trcPlt.plotLines(timeBnd,dat_y=data3vars, ls_modelName = line3names, colors=colors3var, xylabel = xyLabels, titleName="Temporal dynamic of variance", outFig=outPath)  
        #--- figure-2: Proportion dynamics of each component  ----------------------------------------------------------------------
        if iVeg is None:
            outPath = self.figPath + "figure_1-2_dynamic_variation_contribution_"+iThree+".png"
        else:
            outPath = self.figPath + "figure_1-2_dynamic_variation_contribution_"+iVeg+"_"+iThree+".png"
        trcPlt.plotTemFracVC(x, data, colors, labels=lineNames, xyLabels=xyLabels, titleName="Temporal dynamic of variance", outFig=outPath)
        #--- figure-3: the mean and variation of variation contribution of each components ------------------------------------------
        boxData = []; boxPlotX = ["Xp", "GPP", "CUE", "Baseline tau","Temperature", "Precipitation"]
        for i in range(data.shape[0]):
            boxData.append(data[i,:])
        if iVeg is None:
            outPath = self.figPath + "figure_1-3_dynamic_variation_contribution_"+iThree+".png" 
        else:
            outPath = self.figPath + "figure_1-3_dynamic_variation_contribution_"+iVeg+"_"+iThree+".png" 
        trcPlt.plotBoxenplot(boxData, outFig=outPath,titleName="Variation contribution",
                                xlables = boxPlotX,
                                labelxy = ["Traceable components", "Variation contribution"],
                                colors  = colors)

    def runSpatialVariationContribution(self, inFile, nlatlon, iThree="land"):
        data      = pd.read_csv(inFile,header=None).values
        outFig    = self.figPath + "figure_2-1_global_variation_contribution_" + iThree + ".png"
        vMinMax   = [0,6]
        titleName = "The global distribution of the dominant variable"
        unit      = "-"
        colors    = ['#C0C0C0','#32CD32','#98FB98','#F08080','#FFFF00','#1E90FF']
        # my_colors = mpl.colors.ListedColormap(colors, 'indexed')
        trcPlt.plotGlobVC(nlatlon, data, outFig, vMinMax, titleName, unit, figSize=[14,9], mapProj="cyl", cmap=colors)
        #--------------- latitude statistic -------------------------------------------
        latmin, latmax, lonmin, lonmax = nlatlon
        x, y = np.zeros((6,latmax-latmin)), np.linspace(latmin,latmax,latmax-latmin)
        for i in np.arange(6):
            for j in np.arange(latmax-latmin):
                x[i,j] = np.sum(data[j,:]==i+1)
        outFig    = self.figPath + "figure_2-2_global_variation_contribution_latitude_" + iThree + ".png"
        titleName = "The latitude distribution of the dominant variable"
        unit      = "-"
        varNames  = ["C storage potential","GPP","CUE","Baseline tau","Temperature","Precipitation"]
        xylabel   = ["the grid of dominant variable","Latitude"]
        trcPlt.plotGlobLatVC(x,y,varNames,xylabel, titleName,outFig,colors, figsize=[12,9])

    def runSptLatComponents(self, datSpt, datArea, nlatlon, iThree="land"): # var, model, nlat, nlon
        latmin, latmax, lonmin, lonmax = nlatlon
        # xlabels    = {x:   'Carbon storage (kg C m-2)',      xc:    'Carbon storage capacity (kg C m-2)', xp:   'Carbon storage potential (kg C m-2)',
        #               npp: 'NPP (kg C m-2 year-1)',          tau:   'Residence time (year)',              gpp:  'GPP (kg C m-2 year-1)', 
        #               cue: 'CUE (-)',                        bstau: 'Baseline residence time (year)',     senv: 'Environmental scalars (-)',
        #               tas: 'Temperature (degree)',           pr:    'precipitation (mm)'}
        varNames= {x:   'Carbon storage', xc:    'Carbon storage capacity', xp:   'Carbon storage potential',
                   npp: 'NPP',            tau:   'Residence time',          gpp:  'GPP', 
                   cue: 'CUE',            bstau: 'Baseline residence time', senv: 'Environmental scalars',
                   tas: 'Temperature',    pr:    'precipitation'}
        vMinMax = {x:   [0,100], xc:    [0,100],  xp:   [-10,10], npp: [0,10],  tau: [0,1000], gpp: [0,10], 
                   cue: [0,1],   bstau: [0,1000], senv: [0,1],    tas: [0,100], pr:[0,1000]}
        units   = {x:   'kg C m-2', xc:    'kg C m-2', xp: 'kg C m-2', npp: 'kg C m-2 year-1', tau:   'year', gpp:  'kg C m-2 year-1', 
                   cue: '-',        bstau: 'year',     senv: '-',      tas: 'degree',          pr:    'mm'}
        # maskedData = [100000,100000,100000,10000,1,100,100,100,10000,10,10]
        colors     = trcPlt.mkColors(len(self.modelNames))[1:]
        datSpt     = np.where(datArea is np.ma.masked, np.nan, datSpt)
        for key, value in varNames.items():
            dataNew    = datSpt[key,:]   # model, nlat, nlon
            data4plot  = np.where((dataNew<vMinMax[key][0]) | (dataNew>vMinMax[key][1]),np.nan, dataNew)
            datSpt_lat = np.nanmean(data4plot,  axis = 2) # model, nlat
            xyLabel    = [value +"(" +units[key]+")",'Latitude']
            outFig     = self.figPath + "figure_2-3_global_latitude_" + value +"_"+ iThree + ".png"
            trcPlt.plotLatitude(dat_x        = datSpt_lat, # model, nlat
                                dat_y        = np.linspace(latmin,latmax,latmax-latmin),  
                                ls_modelName = self.modelNames,
                                colors       = colors,
                                xylabel      = xyLabel,
                                outFig       = outFig)
    
    def runSptComponentsEvaluation(self, nlatlon, datSpt, iThree="land"):
        # filePath   = r"C:\Users\jzhou\Documents\TraceMe_offline\figures\dataSources\model_results\TraceMe_workdir_historical"
        # modelNames = ['ACCESS-ESM1-5', 'BCC-CSM2-MR','CanESM5','CESM2',
        #             'CNRM-ESM2-1', 'EC-Earth3-Veg','IPSL-CM6A-LR','MIROC-ES2L']
        varNames= {x:   'Carbon storage', xc:    'Carbon storage capacity', xp:   'Carbon storage potential',
                   npp: 'NPP',            tau:   'Residence time',          gpp:  'GPP', 
                   cue: 'CUE',            bstau: 'Baseline residence time', senv: 'Environmental scalars',
                   tas: 'Temperature',    pr:    'precipitation'}
        vMinMax = {x:   [0,100], xc:    [0,100],  xp:   [-10,10], npp: [0,10],  tau: [0,1000], gpp: [0,10], 
                   cue: [0,1],   bstau: [0,1000], senv: [0,1],    tas: [0,100], pr:[0,1000]}
        units   = {x:   'kg C m-2', xc:    'kg C m-2', xp: 'kg C m-2', npp: 'kg C m-2 year-1', tau:   'year', gpp:  'kg C m-2 year-1', 
                   cue: '-',        bstau: 'year',     senv: '-',      tas: 'degree',          pr:    'mm'}
        latmin, latmax, lonmin, lonmax = nlatlon
        nlat, nlon = latmax-latmin, lonmax-lonmin
        for key, value in varNames.items():
            drawData = np.std(datSpt[key,:],axis=0)
            outFig   = self.figPath + "figure_2-4-global_std_" + value + "_" + iThree + ".png"
            titleName = "The standard of " + value
            trcPlt.plotSpatialMap(nLatLon=[latmin, latmax, lonmin, lonmax],drawData=drawData,outFig=outFig,
                                  vMinMax=vMinMax[key],titleName=titleName,unit=units[key])
    
    def runVegTypeVariationContribution(self, inFile, vegType, nlatlon, iThree="land"):
        data   = pd.read_csv(inFile,header=None).values # variation contribution data
        lsRes  = self.func_calVegType(vegType, data, nlatlon)
        dat_ENF, dat_EBF, dat_DNF, dat_DBF, dat_MF, dat_Shrub, dat_Sav, dat_Grass, dat_Tundra, dat_Barren = lsRes
        #--------------------------------------------------------------------------------------------------------
        dataBr = np.zeros((6,10))
        for i in range(6):
            dataBr[i,0] = np.sum(dat_ENF    == i+1); dataBr[i,1] = np.sum(dat_EBF    == i+1)
            dataBr[i,2] = np.sum(dat_DNF    == i+1); dataBr[i,3] = np.sum(dat_DBF    == i+1)
            dataBr[i,4] = np.sum(dat_MF     == i+1); dataBr[i,5] = np.sum(dat_Shrub  == i+1)
            dataBr[i,6] = np.sum(dat_Sav    == i+1); dataBr[i,7] = np.sum(dat_Grass  == i+1)
            dataBr[i,8] = np.sum(dat_Tundra == i+1); dataBr[i,9] = np.sum(dat_Barren == i+1)
        for i in range(10):
            dataBr[:,i] = (dataBr[:,i]/np.sum(dataBr[:,i]))*100
        x          = np.array([1,2,3,4,5,6,7,8,9,10])
        colors     = ['#C0C0C0','#32CD32','#98FB98','#F08080','#FFFF00','#1E90FF']
        ecoLabels  = ['ENF','EBF','DNF','DBF','MF','Shrub','Sav','Grass','Tundra','Barren']
        components = ["C storage potential","GPP","CUE","Baseline tau","Temperature","Precipitation"]
        xyLabel    = ["Vegetation Type","Fraction of dominant variable (%)"]
        outFig     = self.figPath + "figure_3-different_vegType_proportion"+ iThree +".png"
        trcPlt.plotBarEcoVC(x,dataBr,colors,ecoLabels,xyLabel,components,outFig)
    
    def runVegTypeVariationContribution(self, inFiles, iThree = "land"):
        for key, iFile in inFiles.items():
            ########  muti-pie: variation decomposition ############ 
            data       = pd.read_csv(iFile,index_col=False).iloc[:,1].values
            titleName  = "Variation contribution of traceable components\n (" + key + ": " + str(self.timeBnd[0]) + "-" + str(self.timeBnd[1]) + ")"
            components = ["Carbon storage potential", "Carbon storage capacity", "NPP","Residence time","GPP", "CUE", "Baseline residence time",
                        "Temperature", "Precipitation"]
            ##             xp   , xc, npp, tau, gpp, cue, base, T, W 
            newColors  = ['silver', 'chocolate','green',"orange","limegreen","palegreen","lightcoral","yellow","dodgerblue"]
            outFig     = self.figPath+ "figure_4-1_multiPie_vegetationTypes_"+key + "_" +iThree+".png" 
            trcPlt.plotMultPie(np.array(data),labels=components,colors=newColors,titleName=titleName,outFig=outFig)

    def runVegTypeComponentsEvaluation(self, datVT, vegTypes, iThree="land"): # datVT: Var, nVeg, nModel, nTime
        colors     = trcPlt.mkColors(len(self.modelNames))[1:]
        for idxVeg, iVeg in enumerate(vegTypes):
            datTem = datVT[:, idxVeg, :, :]
            # -------- figure-1-4: xc, xp and x -----------
            xylabel = ["Year","Carbon storage and capacity \n(Pg C)"]
            outFig  = self.figPath + "figure_3-1_differentVegetation_xc_xp_x_" + iVeg +"_"+ iThree + ".png"
            trcPlt.plotLines(range_x=self.timeBnd, dat_y=datTem[x,:,:], ls_modelName=self.modelNames,
                            colors=colors, xylabel=xylabel, outFig=outFig, dat_yy=datTem[xc,:,:])
            # -------- figure-1-5: npp-tau ----------------
            xylabel = ["NPP (Pg C/year)","Residence time \n(year)"]
            outFig  = self.figPath + "figure_3-2_differentVegetation_npp_tau_xc_" + iVeg +"_"+ iThree + ".png"
            trcPlt.plotScatter2D(dat_x=datTem[npp,:,:],dat_y=datTem[tau,:,:], ls_modelName=self.modelNames, colors=colors, xylabel=xylabel, outFig=outFig, Zway="x*y")
            # -------- figure-1-6: gpp-npp-cue ------------
            xylabel = ["NPP (Pg  C/year)","GPP (Pg C/year)"]
            outFig  = self.figPath + "figure_3-3_differentVegetation_gpp_npp_cue_" + iVeg +"_"+ iThree + ".png"
            trcPlt.plotScatter2D(dat_x=datTem[npp,:,:], dat_y=datTem[gpp,:,:], ls_modelName=self.modelNames, colors=colors, xylabel=xylabel, outFig=outFig, Zway="x/yD")
            # -------- figure-1-7: tau-baselineTau-SEnv ---
            xylabel = ["Baseline residence time (year)", "Environmental scalars"]
            outFig  = self.figPath + "figure_3-4_differentVegetation_bas_env_tau" + iVeg +"_"+ iThree + ".png"
            trcPlt.plotScatter2D(dat_x=datTem[bstau,:,:], dat_y=datTem[senv,:,:], ls_modelName=self.modelNames, colors=colors, xylabel=xylabel, outFig=outFig, Zway="x/y")
            # -------- figure-1-8: 
            xylabel      = ["Temperature (degree)","Precipitation (mm)"]
            outFig  = self.figPath + "figure_3-5_differentVegetation_tem_pr_" + iVeg +"_"+ iThree + ".png"
            trcPlt.plotScatter2D(dat_x=datTem[tas,:,:], dat_y=datTem[pr,:,:], ls_modelName=self.modelNames, colors=colors, xylabel=xylabel, outFig=outFig)

    # def runVegTypeComponentsEvaluationSpt(self, datSpt, vegType, iThree="land"):# nVar, nModel, nlat, nlon
    #     varNames= {x:   'Carbon storage', xc:    'Carbon storage capacity', xp:   'Carbon storage potential',
    #                npp: 'NPP',            tau:   'Residence time',          gpp:  'GPP', 
    #                cue: 'CUE',            bstau: 'Baseline residence time', senv: 'Environmental scalars',
    #                tas: 'Temperature',    pr:    'precipitation'}
    #     vMinMax = {x:   [0,100], xc:    [0,100],  xp:   [-10,10], npp: [0,10],  tau: [0,1000], gpp: [0,10], 
    #                cue: [0,1],   bstau: [0,1000], senv: [0,1],    tas: [0,100], pr:[0,1000]}
    #     units   = {x:   'kg C m-2', xc:    'kg C m-2', xp: 'kg C m-2', npp: 'kg C m-2 year-1', tau:   'year', gpp:  'kg C m-2 year-1', 
    #                cue: '-',        bstau: 'year',     senv: '-',      tas: 'degree',          pr:    'mm'}
    #     latmin, latmax, lonmin, lonmax = nlatlon
    #     nlat, nlon = latmax-latmin, lonmax-lonmin
    #     # for key, value in varNames.items():
    #     #     drawData = np.std(datSpt[key,:],axis=0)
    #     #     outFig   = self.figPath + "figure_2-4-global_std_" + value + "_" + iThree + ".png"
    #     #     titleName = "The standard of " + value
    #     #     trcPlt.plotSpatialMap(nLatLon=[latmin, latmax, lonmin, lonmax],drawData=drawData,outFig=outFig,
    #     #                           vMinMax=vMinMax[key],titleName=titleName,unit=units[key])
        
    #     # x, xc, xp
    #     datSpt[x, :]


    def func_calVegType(self, datVegType, data4r, nlatlon):
        latmin, latmax, lonmin, lonmax = nlatlon
        nlat, nlon = latmax-latmin, lonmax-lonmin
        datVegType = np.reshape(datVegType, nlat*nlon)
        data4r     = np.reshape(data4r,     nlat*nlon)
        # ----------------------------------------------------------------------------
        # ENF
        dat_ENF_1 = np.where(datVegType == 1, data4r, np.nan)
        idx       = np.where(~np.isnan(dat_ENF_1))
        dat_ENF   = dat_ENF_1[idx]
        # EBF
        dat_EBF_2 = np.where(datVegType == 2, data4r, np.nan)
        idx       = np.where(~np.isnan(dat_EBF_2))
        dat_EBF   = dat_EBF_2[idx]
        # DNF
        dat_DNF_3 = np.where(datVegType == 3, data4r, np.nan)
        idx       = np.where(~np.isnan(dat_DNF_3))
        dat_DNF   = dat_DNF_3[idx]
        # DBF
        dat_DBF_4 = np.where(datVegType == 4, data4r, np.nan)
        idx       = np.where(~np.isnan(dat_DBF_4))
        dat_DBF   = dat_DBF_4[idx]
        # MF
        dat_MF_5 = np.where(datVegType == 5, data4r, np.nan)
        idx       = np.where(~np.isnan(dat_MF_5))
        dat_MF   = dat_MF_5[idx]

        dat_Shrub_67 = np.where((datVegType == 6) | (datVegType ==7), data4r, np.nan)
        idx          = np.where(~np.isnan(dat_Shrub_67))
        dat_Shrub    = dat_Shrub_67[idx]

        dat_Sav8  = np.where((datVegType == 8) | (datVegType == 9), data4r, np.nan)
        idx       = np.where(~np.isnan(dat_Sav8))
        dat_Sav   = dat_Sav8[idx]

        dat_Grass10 = np.where(datVegType == 10, data4r, np.nan)
        idx         = np.where(~np.isnan(dat_Grass10))
        dat_Grass   = dat_Grass10[idx]

        dat_Tundra15 = np.where(datVegType==15,data4r,np.nan)
        idx          = np.where(~np.isnan(dat_Tundra15))
        dat_Tundra   = dat_Tundra15[idx]

        dat_Barren16 = np.where(datVegType == 16, data4r, np.nan)
        idx          = np.where(~np.isnan(dat_Barren16))
        dat_Barren   = dat_Barren16[idx]
        return dat_ENF, dat_EBF, dat_DNF, dat_DBF, dat_MF, dat_Shrub, dat_Sav, dat_Grass, dat_Tundra, dat_Barren