suppressMessages(library(readxl))
suppressMessages(library(vegan))
suppressMessages(library(hier.part))
suppressMessages(library(gtools))
suppressMessages(library(rdaenvpart))
options (warn = -1)
# print("Start to run AnnualTAT_hier_part ...")
args          = commandArgs(T)
inFileName    = args[1]             # inputData
num_scenarios = as.integer(args[2]) # numbers of model or scenarios
outFileName   = args[3]             # output file

# read data
data<- read.csv(file=inFileName, encoding= 'uft-8',header=FALSE, sep=",")
data_org = as.matrix(data)
# different components 
xs  = data_org[1, 1:num_scenarios]
xc  = data_org[2, 1:num_scenarios] 
xp  = data_org[3, 1:num_scenarios] 
npp = data_org[4, 1:num_scenarios] 
res = data_org[5, 1:num_scenarios]
gpp = data_org[6, 1:num_scenarios] 
cue = data_org[7, 1:num_scenarios] 
bas = data_org[8, 1:num_scenarios] 
tem = data_org[9, 1:num_scenarios] 
pre = data_org[10,1:num_scenarios] 

xsVeg   = data_org[11, 1:num_scenarios]
xsSoil  = data_org[12, 1:num_scenarios]
xcVeg   = data_org[13, 1:num_scenarios]
xcSoil  = data_org[14, 1:num_scenarios]
xpVeg   = data_org[15, 1:num_scenarios]
xpSoil  = data_org[16, 1:num_scenarios]
resVeg  = data_org[17, 1:num_scenarios]
resSoil = data_org[18, 1:num_scenarios]
basVeg  = data_org[19, 1:num_scenarios]
basSoil = data_org[20, 1:num_scenarios]

tryCatch({
   # step1 --- carbon storage : carbon storage capacity and carbon storage potential
   step1 = rbind(xs,xc,xp)
   xpc   = data.frame(xc,-xp)
   res1  = hier.part(xs,xpc,family = 'gaussian',gof = "Rsqu", barplot = FALSE)
   step1_xc_x = res1$I.perc[1,1] # carbon storage capacity 
   step1_xp_x = res1$I.perc[2,1] # carbon storage potential

   # step1-1: xp -> xp_veg and xp_soil
   step1_1 = rbind(xp,xpVeg,xpSoil)
   xpDiff  = data.frame(xpVeg,xpSoil)
   res1_1     = hier.part(xp,xpDiff,family = 'gaussian',gof = "Rsqu", barplot = FALSE)
   step1_1_xpVeg_xp  = res1_1$I.perc[1,1] # vegetation carbon storage potential 
   step1_1_xpSoil_xp = res1_1$I.perc[2,1] # soil carbon storage potential

   # step1-2: xc -> xc_veg and xc_soil
   step1_2 = rbind(xc,xcVeg,xcSoil)
   xcDiff  = data.frame(xcVeg,xcSoil)
   res1_2     = hier.part(xc,xcDiff,family = 'gaussian',gof = "Rsqu", barplot = FALSE)
   step1_2_xcVeg_xc  = res1_2$I.perc[1,1] # vegetation carbon storage potential 
   step1_2_xcSoil_xc = res1_2$I.perc[2,1] # soil carbon storage potential

   # step2 --- carbon storage capacity : NPP and residence time
   xc_ln  = log(xc)
   npp_ln = log(npp)
   res_ln = log(res)
   npp_res_ln   = data.frame(npp_ln,res_ln)
   res2         = hier.part(xc_ln, npp_res_ln,family = 'gaussian',gof = "Rsqu", barplot = FALSE )
   step2_npp_xc = res2$I.perc[1,1] # NPP
   step2_res_xc = res2$I.perc[2,1] # Tau

   # step2-1 ----- residence time: vegetation and soil
   step2_1 = rbind(res,resVeg,resSoil)
   resDiff  = data.frame(resVeg,resSoil)
   res2_1     = hier.part(res,resDiff,family = 'gaussian',gof = "Rsqu", barplot = FALSE)
   step2_1_resVeg_res  = res2_1$I.perc[1,1] # vegetation carbon storage potential 
   step2_1_resSoil_res = res2_1$I.perc[2,1] # soil carbon storage potential

   # step3 --- NPP : CUE and GPP
   gpp_ln = log(gpp)
   cue_ln = log(cue)
   gpp_cue_ln = data.frame(gpp_ln,cue_ln)
   res3       = hier.part(npp_ln, gpp_cue_ln,family = 'gaussian',gof = "Rsqu", barplot = FALSE )
   step3_gpp_npp = res3$I.perc[1,1]
   step3_cue_npp = res3$I.perc[2,1]

   # step4 --- tau : baseline tau, temperature and precipitation
   bas_ln =  log(bas)
   tem_ln = -log(tem)
   pre_ln = -log(pre)
   bas_envs_ln = data.frame(bas_ln,tem_ln, pre_ln)
   res4  = hier.part(res_ln, bas_envs_ln,family = 'gaussian',gof = "Rsqu", barplot = FALSE )
   step4_bas_res = res4$I.perc[1,1]
   step4_tem_res = res4$I.perc[2,1]
   step4_pre_res = res4$I.perc[3,1]

   # step4-1 ----------- baseline tau: veg and soil
   step4_1    = rbind(bas,basVeg,basSoil)
   basDiff    = data.frame(basVeg,basSoil)
   res4_1     = hier.part(bas,basDiff,family = 'gaussian',gof = "Rsqu", barplot = FALSE)
   step4_1_basVeg_bas  = res4_1$I.perc[1,1] # vegetation carbon storage potential 
   step4_1_basSoil_bas = res4_1$I.perc[2,1] # soil carbon storage potential

   # total
   cv_gpp = (((step3_gpp_npp/100)*step2_npp_xc)/100)*step1_xc_x
   cv_cue = (((step3_cue_npp/100)*step2_npp_xc)/100)*step1_xc_x
   cv_bas = (((step4_bas_res/100)*step2_res_xc)/100)*step1_xc_x
   cv_tem = (((step4_tem_res/100)*step2_res_xc)/100)*step1_xc_x
   cv_pre = (((step4_pre_res/100)*step2_res_xc)/100)*step1_xc_x

   rp_npp = (step2_npp_xc/100)*step1_xc_x
   rp_res = (step2_res_xc/100)*step1_xc_x

   ab_xpVeg  = (step1_1_xpVeg_xp/100)* step1_xp_x
   ab_xpSoil = (step1_1_xpSoil_xp/100)*step1_xp_x
 
   ab_xcVeg  = (step1_2_xcVeg_xc/100)*step1_xc_x
   ab_xcSoil = (step1_2_xcSoil_xc/100)*step1_xc_x

   ab_resVeg  = (step2_1_resVeg_res/100)*rp_res
   ab_resSoil = (step2_1_resSoil_res/100)*rp_res

   ab_basVeg = (step4_1_basVeg_bas/100)*cv_bas
   ab_basSoil = (step4_1_basSoil_bas/100)*cv_bas

   # if(num_scenarios<3){
   #    data4plot <- c(0,0,0,0,0,0)
   # }else{
   #    data4plot <- c(step1_xp_x,cv_gpp, cv_cue, cv_bas, cv_tem, cv_pre)
   # }

}, error = function(e){
   step1_xp_x = 0
   step1_xc_x = 0
   rp_npp     = 0
   rp_res     = 0
   cv_gpp     = 0
   cv_cue     = 0
   cv_bas     = 0
   cv_tem     = 0
   cv_pre     = 0

   ab_xpVeg   = 0
   ab_xpSoil  = 0
   ab_xcVeg   = 0
   ab_xcSoil  = 0
   ab_resVeg  = 0
   ab_resSoil = 0
   ab_basVeg  = 0
   ab_basSoil = 0
   })

# write results
# max_data = max(data4plot)

res4cv        <- c(step1_xp_x, ab_xpVeg,  ab_xpSoil,  step1_xc_x,  ab_xcVeg,  ab_xcSoil,    rp_npp,      rp_res,      ab_resVeg,   ab_resSoil,   cv_gpp,       cv_cue,        cv_bas, ab_basVeg, ab_basSoil,          cv_tem,        cv_pre)
names(res4cv) <- c("cv_xpInX","ab_xpVeg","ab_xpSoil", "cv_xcInX", "ab_xcVeg", "ab_xcSoil","cv_nppInXc", "cv_tauInXc","ab_resVeg", "ab_resSoil", "cv_gppInAll", "cv_cueInAll", "cv_basTauInAll","ab_basVeg", "ab_basSoil", "cv_temInAll", "cv_preInAll")
# res_cv <- c(step1_xc_x,step1_xp_x,step2_npp_xc,step2_res_xc,step3_gpp_npp,step3_cue_npp,step4_bas_res,step4_tem_res,step4_pre_res)
write.csv(res4cv, file =outFileName)