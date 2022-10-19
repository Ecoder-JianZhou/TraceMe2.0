import matplotlib.pyplot as plt
from pyecharts.charts import Sankey
from pyecharts import options as opts
import seaborn as sns
import numpy as np
from mpl_toolkits.basemap import Basemap, cm
import matplotlib as mpl
import math, random

def randColor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def mkColors(ncolor):
    # colors = ['#black', 
    #           '#red', '#orange','#yellow', '#green', '#blue','#indugo', '#purple', 
    #           '#lightcoral','#darkorange', '#greenyellow', '#lightgreen','#cyan','#cornflowerblue','#violet',
    #           '#saddlebrown', '#lawngreen', '#darksage','#dodgerblue', '#pink']
    colors = ['#000000', 
              '#FF0000', '#FFA500','#FFFF00', '#008000', '#0000FF','#4B0082', '#800080', 
              '#F08080', '#FF8C00', '#ADFF2F', '#90EE90','#00FFFF','#6495ED','#EE82EE',
              '#8B4513', '#7CFC00', '#3CB371','#1E90FF', '#FFC0CB']
    if ncolor > 20:
        retColors = colors
        while len(colors) < ncolor:
            i_color = randColor()
            if i_color not in colors:
                retColors.append(i_color)
    else:
        retColors = colors[:ncolor+1]
    return retColors

def convertLon(data, nLatLon):
    latmin,latmax,lonmin,lonmax = nLatLon # nLatLon: [latmin,latmax,lonmin,lonmax]
    #if latmax - latmin <= 2: 
    nlat,nlon     = latmax-latmin,lonmax-lonmin
    dataCovert     = np.full([nlat,nlon],np.nan) #drawData
    if (lonmax - lonmin)%2 == 1: 
        midInd = (lonmax-lonmin)/2
        dataCovert[:,:midInd] = data[:,midInd+1:]
        dataCovert[:,midInd+1:] = data[:,:midInd]
        dataCovert[:,midInd]  = data[:,midInd]
        del data
    else:
        midInd = (lonmax-lonmin)//2
        dataCovert[:,:midInd] = data[:,midInd:]
        dataCovert[:,midInd:] = data[:,:midInd]
        del data
    return dataCovert

def plotMultPie(data, labels, colors, titleName, outFig, figSize=[10,8]):
    fig  = plt.figure(figsize=(figSize[0],figSize[1])) # figsize
    ax   = fig.add_axes([-0.05,0.05,0.86,0.86])        # [left, bottom, width, height]
    size = 0.3
    for spines in ax.spines.values(): # spines: the axes
        spines.set_visible(True)      # the value on axes 
    ##### data #####
    vals_1 = data[:2]                 # Xp and Xc
    vals_2 = data[[0,2,3]]            # Xp, NPP and Tau
    vals_3 = data[[0,4, 5, 6, 7, 8]]  # Xp, GPP, CUE, Baseline Tau, Temperature and Precipitation
    ##### colors #####
    inner_colors = colors[:2] #['silver', 'chocolate'] #colors[:2]
    mid_colors   = [colors[0], colors[2], colors[3]] #['silver', 'green',"orange"] #[colors[0]].append(colors[2:4])
    outer_colors = [colors[0], colors[4], colors[5], colors[6], colors[7], colors[8]] #['silver', "limegreen","palegreen","lightcoral","yellow","dodgerblue"] #[colors[0]].append(colors[4:])
    ##### plot ... #####
    wedges, texts= ax.pie(data, radius = 0.395, colors=colors, startangle = 0) # just for legend
    wedges1, texts1, autotexts1 = ax.pie(vals_1, radius = 0.395, colors=inner_colors, startangle = 0, 
        autopct   = "%3.1f%%", textprops = dict(color="k"), 
        wedgeprops=dict( edgecolor='w')) # carbon storage capacity and potential
    wedges2, texts2, autotexts2 = ax.pie(vals_2, radius = 0.7, colors=mid_colors, startangle = 0, pctdistance=0.395+0.35,
        autopct   = "%3.1f%%", textprops = dict(color="k"),
        wedgeprops=dict(width=size, edgecolor='w')) # NPP Tau and potential
    wedges3, texts3, autotexts3 = ax.pie(vals_3, radius=1, colors=outer_colors, startangle = 0, pctdistance=0.7+0.15,
        autopct   = "%3.1f%%", textprops = dict(color="k"),
        wedgeprops=dict(width=size, edgecolor='w')) # GPP CUE Baseline_Tau and Temperature precipitation potential 
    plt.legend(wedges, labels,fontsize=14,loc="lower left",bbox_to_anchor=(0.9, 0, 0.34, 1)) # x0,y0,width,height
    plt.setp(autotexts1, size=14, weight="bold")
    plt.setp(autotexts2, size=14, weight="bold")
    plt.setp(autotexts3, size=14, weight="bold")
    plt.setp(texts1, size=16)
    plt.setp(texts2, size=16)   
    plt.setp(texts3, size=16)
    plt.title(titleName,fontsize=20, y=0.95)    
    plt.savefig(outFig)#,bbox_inches='tight')

def plotSankey(nodes, links, colors, titleName, outFigHtml):
    pic = (Sankey(init_opts=opts.InitOpts(width="600px", height="400px")) #设置图表的宽度和高度
           # .set_colors(colors)
            .add(
                "Results",
                nodes,#读取节点
                links,#读取路径
                linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),#设置线条样式
                label_opts=opts.LabelOpts(position="inside"),#设置标签配置项
                node_align=str( "justify"),#设置节点对齐方式：right，left,justify(节点双端对齐)
                orient="horizontal", 
                pos_top="10%",
                node_width=30,
                node_gap = 20,
                layout_iterations=64,
                is_draggable=True, # 是否可以拖拽
                )
            .set_global_opts(title_opts=opts.TitleOpts(title=titleName))#表名
          )
    pic.render(outFigHtml)

def plotHeatmap(data, outFig, titleName, vminmax, figSize=[9,9]):
    fig = plt.figure(figsize=(figSize[0],figSize[1]))
    sns.heatmap(data, linewidths = 0.1, vmin=vminmax[0], vmax = vminmax[1],cmap='bwr',square= True)
    plt.title(titleName,fontsize=20)
    plt.tick_params(labelsize=16)
    plt.savefig(outFig, bbox_inches='tight')

def plotTemFracVC(x, data, colors, labels, xyLabels, titleName, outFig, figSize = (12,9)):
    '''
     x   : time 
     data: [components,  time-series]
    '''
    fig  = plt.figure(figsize=(figSize[0],figSize[1]))
    ax   = fig.add_axes([0.20,0.16,0.75,0.78])
    data4plot = (data/data.sum(axis=0))*100
    ntime = data.shape[1]
    for i in range(6):
        if i == 0:
            dat_y  = data4plot[0,:]
            dat_yy = np.zeros((ntime)) 
        elif i == 1:
            dat_y  = data4plot[:2,:].sum(axis=0)
            dat_yy = data4plot[0,:]
        else:
            dat_y  = data4plot[:i+1,:].sum(axis=0)
            dat_yy = data4plot[:i,:].sum(axis=0)
        plt.fill_between(x, dat_yy, dat_y, facecolor=colors[i])#,alpha=0.3)
    plt.xlim(min(x),max(x))
    plt.ylim(0,100)
    fontsize = 25
    plt.xlabel(xlabel=xyLabels[0],fontsize=fontsize)
    plt.ylabel(ylabel=xyLabels[1],fontsize=fontsize)
    plt.tick_params(labelsize=22)
    plt.title(titleName, fontsize=fontsize)
    plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0. , fancybox=True, ncol=1, labels=labels, fontsize=18)
    plt.savefig(outFig, bbox_inches='tight')

def plotBoxenplot(data,outFig,titleName, xlables, labelxy, colors,figSize=[9,9]):
    fontsize = 25
    fig, ax = plt.subplots(figsize=(figSize[0],figSize[1]))
    bplot   = ax.boxplot(data,
                        vert=True,  # vertical box alignment
                        patch_artist=True,  # fill with color
                        labels= xlables)  # will be used to label x-ticks
    # for bplot in boxplot:
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title(titleName, fontsize=fontsize)
    ax.set_xlabel(labelxy[0],fontsize=fontsize)
    ax.set_ylabel(labelxy[1],fontsize=fontsize)
    plt.xticks(rotation=70,fontsize=22)
    plt.yticks(fontsize=20)
    plt.savefig(outFig, bbox_inches='tight')

def plotLines(range_x, dat_y, ls_modelName, colors, xylabel, outFig, titleName = None, dat_yy=None, figsize=[12,9]):       
    fontsize = 25
    x = range(range_x[0],range_x[1])
    fig = plt.figure(figsize=(figsize[0],figsize[1]))
    ax  = fig.add_axes([0.20,0.16,0.75,0.78])
    for i, modelName in enumerate(ls_modelName):
        plt.plot(x, dat_y[i,:], colors[i], linewidth=2, label = modelName)
        if dat_yy is not None:
            plt.fill_between(x,dat_yy[i,:],dat_y[i,:],facecolor=colors[i],alpha=0.3)
    if range_x[1] - range_x[0]>6:
        new_ticks = np.linspace(range_x[0]-1,range_x[1],6).astype(int)
    else:
        new_ticks = np.arange(range_x[0]-1,range_x[1])
    plt.xlim(range_x[0],range_x[1])
    plt.xticks(new_ticks)
    plt.xlabel(xylabel[0],fontsize=fontsize)
    plt.ylabel(xylabel[1],fontsize=fontsize)
    plt.tick_params(labelsize=22)
    if titleName is not None:
        plt.title(titleName, fontsize=fontsize)
    #plt.legend(loc='right')#, bbox_to_anchor=(0,0),ncol=1, borderaxespad = 0.,fontsize=18)
    plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0., fontsize=18)
    plt.savefig(outFig, bbox_inches='tight')

def plotBox(data,outFig,titleName, xlables, labelxy, colors,figSize=[9,9]):
    fontsize = 25
    fig, ax = plt.subplots(figsize=(figSize[0],figSize[1]))
    bplot   = ax.boxplot(data,
                        vert=True,  # vertical box alignment
                        patch_artist=True,  # fill with color
                        labels= xlables)  # will be used to label x-ticks
    # for bplot in boxplot:
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title(titleName, fontsize=fontsize)
    ax.set_xlabel(labelxy[0],fontsize=fontsize)
    ax.set_ylabel(labelxy[1],fontsize=fontsize)
    plt.xticks(rotation=70,fontsize=22)
    plt.yticks(fontsize=20)
    plt.savefig(outFig, bbox_inches='tight')

def plotScatter2D(dat_x, dat_y, ls_modelName,colors,xylabel,outFig, vMinMax=None,figSize=[9,7], Zway = None):
    """ Zway : "x*y"; "y/x","x/y","y/xD","x/yD". """
    fontsize = 25; lsZway = ["x*y","y/x","x/y","y/xD","x/yD"]
    fig = plt.figure(figsize=(figSize[0],figSize[1]))
    ax  = fig.add_axes([0.20,0.16,0.75,0.78])
    #ax.set_position([box.x0, box.y0, box.width* 0.8 , box.height])
    ### contour ####
    if Zway is not None:
        n = 256
        if np.max(dat_x) - np.min(dat_x) < 1:
            x_min = np.min(dat_x) - (np.max(dat_x)-np.min(dat_x)) * 0.1
            x_max = np.max(dat_x) + (np.max(dat_x)-np.min(dat_x)) * 0.1
        else:
            x_min, x_max  = int(np.min(dat_x))-1, math.ceil(np.max(dat_x))+1  

        if np.max(dat_y) - np.min(dat_y) < 1:
            y_min = np.min(dat_y) - (np.max(dat_y)-np.min(dat_y)) * 0.1
            y_max = np.max(dat_y) + (np.max(dat_y)-np.min(dat_y)) * 0.1
        else:
            y_min, y_max  = int(np.min(dat_y))-1, math.ceil(np.max(dat_y))+1

        x = np.linspace(x_min, x_max, n)
        y = np.linspace(y_min, y_max, n)
        X, Y = np.meshgrid(x, y)
        if Zway in lsZway:
            if Zway == "x*y":
                Z = X * Y
                Z = np.around(Z, decimals = 0)
            elif Zway == "y/x":
                Z = Y/X
                Z = np.around(Z, decimals = 0)
            elif Zway == "x/y":
                Z = X/Y
                Z = np.around(Z, decimals = 0)
            elif Zway == "y/xD":
                x = np.linspace(np.min(dat_x), np.max(dat_x), n) # x = np.linspace(25, 75, n)
                y = np.linspace(np.min(dat_y), np.max(dat_y), n) # y = np.linspace(75, 145, n)
                X, Y = np.meshgrid(x, y)
                Z = Y/X
                Z = np.around(Z, decimals = 3)
            elif Zway == "x/yD":
                x = np.linspace(np.min(dat_x), np.max(dat_x), n) # x = np.linspace(25, 75, n)
                y = np.linspace(np.min(dat_y), np.max(dat_y), n) # y = np.linspace(75, 145, n)
                X, Y = np.meshgrid(x, y)
                Z = X/Y
                Z = np.around(Z, decimals = 3)
            C = ax.contour(X, Y, Z, 8, colors = "black", linestyles="--", alpha = 0.3)
            ax.clabel(C, inline=True, fontsize=12, fmt='%.2f')
        else:
            print("Your Zway must be defined as: x*y; y/x; x/y; y/xD; x/yD; ")
    n_model=0
    for modelName in ls_modelName:
        plt.scatter(dat_x[n_model,:], dat_y[n_model,:], c=colors[n_model],marker='o',s=150,label=modelName,alpha=0.3, linewidth=0)
        n_model=n_model+1
    # if vMinMax is not None:
    #     xVmin, xVmax, yVmin, yVmax = vMinMax
    #     if xVmin < 
    #     plt.xlim()
    plt.xlabel(xlabel=xylabel[0],fontsize=fontsize)
    plt.ylabel(ylabel=xylabel[1],fontsize=fontsize)
    plt.tick_params(labelsize=22)
    #plt.legend(loc='center left', bbox_to_anchor=(0.59,0.5),ncol=1, borderaxespad = 0.,fontsize=18)
    plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig(outFig, bbox_inches='tight')

def plotGlobVC(nLatLon, drawData, outFig, vMinMax, titleName, unit, figSize=[14,9], mapProj="cyl", cmap="jet"): 
    '''
        just for global
    '''
    latmin,latmax,lonmin,lonmax = -90,90,-180,180
    r_lat = np.arange(latmin,latmax)
    r_lon = np.arange(lonmin,lonmax)

    fig   = plt.figure(figsize=(figSize[0],figSize[1]))
    ax    = fig.add_axes([0.05,0.15,0.9,0.8])
    axPie = fig.add_axes([0.04, 0.3, 0.2, 0.2])
    m    = Basemap(projection = mapProj, resolution='l', ax=ax)
    x, y = m(*np.meshgrid(r_lon,r_lat))

    lim = np.linspace(0,6,7)
    my_colors = mpl.colors.ListedColormap(cmap, 'indexed')
    color4map = my_colors
    ctf = ax.contourf(x,y,drawData.squeeze(),lim,cmap=color4map,zorder=1,extend="both")
    m.drawcoastlines(linewidth=0.4,zorder=3)
    m.drawlsmask(land_color=(0, 0, 0, 0), ocean_color="w",lakes=True, zorder=2)
    ### add_axes for pie plot ###
    weight = np.zeros((6))
    for i in range(6):
        weight[i]= np.sum(drawData==i+1)
    print(weight,cmap)
    wedges, texts, autotexts = axPie.pie(weight,
                                    autopct   = "%3.1f%%", # number format
                                    textprops = dict(color="w"), # text info
                                    colors    = cmap)
    # ### legend ... ###
    # cbar_ax2 = fig.add_axes([0.08, 0.27, 0.05, 0.25])
    # # norm = mpl.colors.BoundaryNorm()
    # cb_2 = fig.colorbar(ctf, cax=cbar_ax2, orientation="vertical")
    # cb_2.set_ticks(np.linspace(0.5, 5.5, 6))
    # # = np.around(np.linspace(lim_min, lim_max, 6),decimals=0)
    # # cb_t = cb_t.astype(np.int16)
    # cb_2.set_ticklabels(['Scalar (precipitation)', 'Scalar (temperature)', 'Baseline residence time', 'CUE', 'GPP','C storage potential'])
    # for l in cb_2.ax.yaxis.get_ticklabels():
    #     l.set_family('Arial')
    #     l.set_size(14)
    position = fig.add_axes([0.17, 0.15, 0.67, 0.03]) # left,bottom,right,top
    cb = plt.colorbar(ctf, cax=position,orientation="horizontal")
    font1 = {'size':40}
    cb.set_ticks(np.linspace(0.5, 5.5, 6),font1)
    cb.set_ticklabels(['C storage potential','GPP','CUE','Baseline tau','S_temperature', 'S_precipitation'])
    cb.set_label(titleName+' ('+unit+")", fontsize=28)
    plt.savefig(outFig) 

def plotSpatialMap(nLatLon, drawData, outFig, vMinMax, titleName, unit, figSize=[14,9], mapProj="cyl", cmap="jet", plotCV=False): 
    latmin,latmax,lonmin,lonmax = nLatLon # nLatLon: [latmin,latmax,lonmin,lonmax]
    #if latmax - latmin <= 2: 
    nlat,nlon     = latmax-latmin,lonmax-lonmin
    midLat,midLon = latmin+(latmax-latmin)/2, lonmin+(lonmax-lonmin)/2
    data4plot     = np.full([nlat,nlon],np.nan) #drawData
    colors4pie = cmap
    # if (lonmax - lonmin)%2 == 1: 
    #     midInd = (lonmax-lonmin)/2
    #     data4plot[:,:midInd] = drawData[:,midInd+1:]
    #     data4plot[:,midInd+1:] = drawData[:,:midInd]
    #     data4plot[:,midInd]  = drawData[:,midInd]
    #     del drawData
    # else:
    #     midInd = (lonmax-lonmin)//2
    #     data4plot[:,:midInd] = drawData[:,midInd:]
    #     data4plot[:,midInd:] = drawData[:,:midInd]
    #     del drawData
    data4plot = drawData
    r_lat = np.arange(latmin,latmax)
    r_lon = np.arange(lonmin,lonmax)

    fig  = plt.figure(figsize=(figSize[0],figSize[1]))
    ax   = fig.add_axes([0.05,0.15,0.9,0.8])
    m    = Basemap(projection = mapProj, resolution='l', lat_0=midLat, lon_0=midLon, llcrnrlon = lonmin, llcrnrlat = latmin, urcrnrlon = lonmax, urcrnrlat = latmax, ax=ax)
    x, y = m(*np.meshgrid(r_lon,r_lat))

    if vMinMax[0]<vMinMax[1]:
        if plotCV:
            lim = np.linspace(0,6,7)
            my_colors = mpl.colors.ListedColormap(cmap, 'indexed')
            cmap = my_colors
        else:
            lim = np.linspace(vMinMax[0], vMinMax[1], 500)
        ctf = ax.contourf(x,y,data4plot.squeeze(),lim,cmap=cmap,zorder=1,extend="both")
    else:
        ctf = ax.contourf(x,y,data4plot.squeeze(),500,cmap=cmap,zorder=1)

    m.drawcoastlines(linewidth=0.4,zorder=3)
    m.drawlsmask(land_color=(0, 0, 0, 0), ocean_color="w",lakes=True, zorder=2)

    if plotCV:
        ### add_axes for pie plot ###
        weight = np.zeros((6))
        for i in range(6):
            weight[i]= np.sum(data4plot==i+1)
        print(weight,cmap)
        pie_ax = fig.add_axes([0.08, 0.27, 0.05, 0.25])
        wedges, texts, autotexts = ax.pie(weight,
                                       autopct   = "%3.1f%%", # number format
                                       textprops = dict(color="w"), # text info
                                       colors    = colors4pie)
        ### legend ... ###
        cbar_ax2 = fig.add_axes([0.08, 0.27, 0.05, 0.25])
        # norm = mpl.colors.BoundaryNorm()
        cb_2 = fig.colorbar(ctf, cax=cbar_ax2, orientation="vertical")
        cb_2.set_ticks(np.linspace(0.5, 5.5, 6))
        # = np.around(np.linspace(lim_min, lim_max, 6),decimals=0)
        # cb_t = cb_t.astype(np.int16)
        cb_2.set_ticklabels(['Scalar (precipitation)', 'Scalar (temperature)', 'Baseline residence time', 'CUE', 'GPP','C storage potential'])
        for l in cb_2.ax.yaxis.get_ticklabels():
            l.set_family('Arial')
            l.set_size(14)
    else:
        position = fig.add_axes([0.17, 0.15, 0.67, 0.03]) # left,bottom,right,top
        cb = plt.colorbar(ctf, cax=position,orientation="horizontal")
        font1 = {'size':40}
        if vMinMax[0] < vMinMax[1]:
            levels1 = np.linspace(vMinMax[0], vMinMax[1], 6)
            cb.set_ticks(levels1,font1)
        cb.set_label(titleName+' ('+unit+")", fontsize=28)
    plt.savefig(outFig) 

def plotGlobLatVC(x,y,ls_modelName,xylabel, titleName,outFig, colors,figsize=[12,9]):
    print("plot latitude cv")
    fontsize = 25
    # x = range(range_x[0],range_x[1])
    fig = plt.figure(figsize=(figsize[0],figsize[1]))
    ax  = fig.add_axes([0.20,0.16,0.75,0.78])
    data4plot = (x/x.sum(axis=0))*100
    print(data4plot)
    for i in range(6):
        if i == 0:
            dat_x  = data4plot[0,:]
            dat_xx = np.zeros((len(y))) 
        elif i == 1:
            dat_x  = data4plot[:2,:].sum(axis=0)
            dat_xx = data4plot[0,:]
        else:
            dat_x  = data4plot[:i+1,:].sum(axis=0)
            dat_xx = data4plot[:i,:].sum(axis=0)
        plt.fill_betweenx(y, dat_xx, dat_x, facecolor=colors[i])#,alpha=0.3)
    limY = np.sum(data4plot,axis=0)
    limYY = np.array([y,limY])
    limYYY = np.where(np.isnan(limYY[1,:]),np.nan,limYY[0,:])
    print(limYY)
    print(limYY.shape)
    print(limYYY)
    print(np.nanmax(limYYY),np.nanmin(limYYY))
    # plt.xticks(new_ticks)
    plt.xlim(0,100)
    plt.ylim(np.nanmin(limYYY),np.nanmax(limYYY))
    plt.xlabel(xylabel[0],fontsize=fontsize)
    plt.ylabel(xylabel[1],fontsize=fontsize)
    plt.tick_params(labelsize=22)
    if titleName is not None:
        plt.title(titleName, fontsize=fontsize)
    #plt.legend(loc='right')#, bbox_to_anchor=(0,0),ncol=1, borderaxespad = 0.,fontsize=18)
    plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0. , fancybox=True, ncol=1, labels=ls_modelName, fontsize=18)
    plt.savefig(outFig, bbox_inches='tight')

def plotBarEcoVC(x,data4bar,colors,ecoLabels,xyLabel,components,outFig):
    plt.bar(x, data4bar[0,:], align="center", color=colors[0], tick_label=ecoLabels, label=components[0])
    for i in range(1,data4bar.shape[0]):
        print(data4bar[i,:])
        print(np.sum(data4bar[i,:]))
        plt.bar(x, data4bar[i,:], align="center", bottom=np.sum(data4bar[:i,:],axis=0), color=colors[i], label=components[i])
    plt.xlabel(xyLabel[0])
    plt.ylabel(xyLabel[1])
    plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig(outFig, bbox_inches='tight')

def plotEcoScatter2D(dat_x, dat_y, ls_modelName,colors,xylabel,outFig, figSize=[9,7], dat_z = None, Zway = "multiply"):
    fontsize = 25
    fig = plt.figure(figsize=(figSize[0],figSize[1]))
    ax  = fig.add_axes([0.20,0.16,0.75,0.78])
    #ax.set_position([box.x0, box.y0, box.width* 0.8 , box.height])
    ### contour ####
    if dat_z is not None:
        n = 256
        if np.max(dat_x) - np.min(dat_x) < 1:
            x_min = np.min(dat_x) - (np.max(dat_x)-np.min(dat_x)) * 0.1
            x_max = np.max(dat_x) + (np.max(dat_x)-np.min(dat_x)) * 0.1
        else:
            x_min, x_max  = int(np.min(dat_x))-1, math.ceil(np.max(dat_x))+1  

        if np.max(dat_y) - np.min(dat_y) < 1:
            y_min = np.min(dat_y) - (np.max(dat_y)-np.min(dat_y)) * 0.1
            y_max = np.max(dat_y) + (np.max(dat_y)-np.min(dat_y)) * 0.1
        else:
            y_min, y_max  = int(np.min(dat_y))-1, math.ceil(np.max(dat_y))+1

        x = np.linspace(x_min, x_max, n)
        y = np.linspace(y_min, y_max, n)
        X, Y = np.meshgrid(x, y)
        if Zway == "multiply":
            Z = X * Y
            Z = np.around(Z, decimals = 0)
        elif Zway == "divide1":
            Z = Y/X
            Z = np.around(Z, decimals = 0)
        elif Zway == "divide2":
            Z = X/Y
            Z = np.around(Z, decimals = 0)
        C = ax.contour(X, Y, Z, 6, colors = "black", linestyles="--", alpha = 0.3)
        ax.clabel(C, inline=True, fontsize=12, fmt='%.0f')

    n_model=0
    for modelName in ls_modelName:
        plt.scatter(dat_x[n_model,:], dat_y[n_model,:], c=colors[n_model],marker='o',s=150,label=modelName,alpha=0.3, linewidth=0)
        n_model=n_model+1
    plt.xlabel(xlabel=xylabel[0],fontsize=fontsize)
    plt.ylabel(ylabel=xylabel[1],fontsize=fontsize)
    plt.tick_params(labelsize=22)
    #plt.legend(loc='center left', bbox_to_anchor=(0.59,0.5),ncol=1, borderaxespad = 0.,fontsize=18)
    plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig(outFig, bbox_inches='tight')

def plotLatitude(dat_x, dat_y, ls_modelName, colors, xylabel, outFig,  figsize=[9,7]):
    fontsize = 25
    fig = plt.figure(figsize=(figsize[0],figsize[1]))
    ax  = fig.add_axes([0.20,0.16,0.75,0.78])
    for i, modelName in enumerate(ls_modelName):
        plt.plot(dat_x[i,:], dat_y, colors[i], linewidth=2, label = modelName)
    plt.yticks(np.linspace(np.min(dat_y),np.max(dat_y),7))
    plt.ylim(-60,80)
    plt.xlabel(xylabel[0],fontsize=fontsize)
    plt.ylabel(xylabel[1],fontsize=fontsize)
    plt.tick_params(labelsize=22)
    #plt.legend(loc='right')#, bbox_to_anchor=(0,0),ncol=1, borderaxespad = 0.,fontsize=18)
    plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig(outFig, bbox_inches='tight')

# # subplot 9: baseline residence time
# ax = plt.subplot(3,3,9)
# #box = ax.get_position()
# #ax.set_position([box.x0, box.y0, box.width, box.height])
# x_6= np.arange(num_models)
# ax.bar(x_6, dat_obj_basedResTime[:, 0], color=models_color)
# ax.tick_params(labelsize=14)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Arial') for label in labels]
# plt.xticks(x_6, np.arange(num_models)+1)
# ax.set_xlabel('Model', font1)
# ax.set_ylabel('Baseline C residence time (year)', font1)
# ax.text(-0.1, 1.05, "(i)", transform=ax.transAxes,fontdict=font1) #x

# font1 = {'family': 'Arial', 'weight': 'normal', 'size': 15}
# patches = [mpatches.Patch(color=models_color[i], label="{:s}".format(list_modelname[i]) ) for i in range(len(models_color)) ]
# plt.legend(handles= patches,loc='lower center', bbox_to_anchor=(-0.75, -0.45), ncol=np.math.ceil(num_models/2), borderaxespad=0,
#            prop=font1, edgecolor='w')
# #fig.tight_layout()
# plt.subplots_adjust(wspace=0.28,hspace=0.25,bottom=0.23)
# fig.align_labels()
# #plt.show()
# plt.savefig("figure1_9_new6.tiff", dpi=600, bbox_inches='tight')