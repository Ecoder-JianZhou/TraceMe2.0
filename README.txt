This is the README for TraceMe 2.0 (offline version)

v2.4.1
modify some plotting bugs:
    20210726: runSptLatComponents
    line 93: "for i in range(datHeatMap.shape[1]):" change to "for i in range(datHeatMap.shape[0]):"

v2.4.0
change the yml file to define the vegetation type and aboveground-belowground analysis
    analysis:
        VegetationType: yes
        aboveAndBelowGround: yes
modify some bugs:
1. yml read: with no .yml can also be read.
2. change the unit change into the preproprocessing processes 
3. write the parallel run to the preproprocessing processes

v2.3.1
1. change "dynamic temporal of traceable components" from 6 variables to 3 variables: npp, tau and carbon potential
2. modify some bugs

v2.3.0
redesign the plotting scripts:
1. temporal scripts:
    total traceability analysis:
        0.1 multi-pie figure
        0.2 uncertainty source of sankey figure
        1.1 std of traceable components
    dynamic temporal:
        1.2 temporal values of traceable components
        1.3 dynamic temporal of traceable components 
        1.4 statistic of traceable components
    traceable components:
        1.5 x-xc-xp
        1.6 npp-tau
        1.7 gpp-cue-npp
        1.8 bstau-env-tau
        1.9 tas-pr
2. spatial scripts:
    2.1 variation contribution 
    2.2 latitude statistic of variation contribution, and traceable components
    2.3 std of spatial components

v2.2.0
1. add the analysis of different vegetation types
2. plotting scripts include: dynamic temporal variation contribution;

v2.1.0
1. use new class to calculate the variation decomposition
2. use a class to define the plotting object
3. divide the cland into three parts: cLand, aboveground carbon and belowground carbon
    aboveground: vegetation (cVeg: cLeaf, cStem and cRoot)
    belowground: cCwd, litter and soil carbon

v2.0.0
re-design the framework of TraceMe package:
1. use the yml file to preset the cases to run the package:
2. Separate the preproprocessing processes and the TraceMe script.
    this class can be used to read other handling script.
3. separate the TraceMe script into: traceable components and variation decomposition
4. new scripts to plot.