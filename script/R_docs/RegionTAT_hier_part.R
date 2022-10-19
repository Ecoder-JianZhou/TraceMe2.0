suppressMessages(library(MASS))
suppressMessages(library(boot))
suppressMessages(library(grid))
suppressMessages(library(Matrix))
suppressMessages(library(survival))
suppressMessages(library(survey))
suppressMessages(library(relaimpo))
suppressMessages(library(readxl))
suppressMessages(library(vegan))
suppressMessages(library(hier.part))
suppressMessages(library(gtools))
suppressMessages(library(rdaenvpart))
#library(reticulate)
suppressMessages(library(foreach))
suppressMessages(library(doParallel))

options (warn = -1)
args          = commandArgs(T)
inFilePath    = args[1]
num_scenarios = as.integer(args[2])
outFileName   = args[3]
latmin        = as.integer(args[4])
latmax        = as.integer(args[5])
lonmin        = as.integer(args[6])
lonmax        = as.integer(args[7])

ls_filename <- c ('baseline_residence_time','carbon_storage_capacity','carbon_storage_potential','carbon_storage','cue','gpp','npp','rain','residence_time','temperature')

fileName = dir(inFilePath)
n_lat    = latmax - latmin
n_lon    = lonmax - lonmin
nlatlon  = n_lat*n_lon

data_bas = array(rep(NaN, nlatlon*num_scenarios), dim=c(nlatlon,num_scenarios));
data_cc  = array(rep(NaN, nlatlon*num_scenarios), dim=c(nlatlon,num_scenarios));
data_cs  = array(rep(NaN, nlatlon*num_scenarios), dim=c(nlatlon,num_scenarios));
data_cp  = array(rep(NaN, nlatlon*num_scenarios), dim=c(nlatlon,num_scenarios));
data_npp = array(rep(NaN, nlatlon*num_scenarios), dim=c(nlatlon,num_scenarios));
data_gpp = array(rep(NaN, nlatlon*num_scenarios), dim=c(nlatlon,num_scenarios));
data_cue = array(rep(NaN, nlatlon*num_scenarios), dim=c(nlatlon,num_scenarios));
data_res = array(rep(NaN, nlatlon*num_scenarios), dim=c(nlatlon,num_scenarios));
data_pre = array(rep(NaN, nlatlon*num_scenarios), dim=c(nlatlon,num_scenarios));
data_tem = array(rep(NaN, nlatlon*num_scenarios), dim=c(nlatlon,num_scenarios));

# Read data
for (i in 1:num_scenarios){
  read_file = paste(inFilePath,"/",ls_filename[1],"_",as.character(i),".csv",sep="")
  csv_bas   = read.csv(file = paste(inFilePath,"/",ls_filename[1],"_",as.character(i),".csv",sep=""), header = F, sep = ',')
  csv_cc    = read.table(file = paste(inFilePath,"/",ls_filename[2],"_",as.character(i),".csv",sep=""),sep=',')
  csv_cp    = read.table(file = paste(inFilePath,"/",ls_filename[3],"_",as.character(i),".csv",sep=""),sep=',')
  csv_cs    = read.table(file = paste(inFilePath,"/",ls_filename[4],"_",as.character(i),".csv",sep=""),sep=',')
  csv_cue   = read.table(file = paste(inFilePath,"/",ls_filename[5],"_",as.character(i),".csv",sep=""),sep=',')
  csv_gpp   = read.table(file = paste(inFilePath,"/",ls_filename[6],"_",as.character(i),".csv",sep=""),sep=',')
  csv_npp   = read.table(file = paste(inFilePath,"/",ls_filename[7],"_",as.character(i),".csv",sep=""),sep=',')
  csv_pre   = read.table(file = paste(inFilePath,"/",ls_filename[8],"_",as.character(i),".csv",sep=""),sep=',')
  csv_res   = read.table(file = paste(inFilePath,"/",ls_filename[9],"_",as.character(i),".csv",sep=""),sep=',')
  csv_tem   = read.table(file = paste(inFilePath,"/",ls_filename[10],"_",as.character(i),".csv",sep=""),sep=',')
  
  ######### for parallel run ##########
  data_bas[1:nlatlon,i] = as.vector(as.matrix(csv_bas)) #as.matrix(csv_bas)
  data_cc[1:nlatlon,i]  = as.vector(as.matrix(csv_cc))
  data_cs[1:nlatlon,i]  = as.vector(as.matrix(csv_cs))
  data_cp[1:nlatlon,i]  = as.vector(as.matrix(csv_cp))
  data_npp[1:nlatlon,i] = as.vector(as.matrix(csv_npp))
  data_gpp[1:nlatlon,i] = as.vector(as.matrix(csv_gpp))
  data_cue[1:nlatlon,i] = as.vector(as.matrix(csv_cue))
  data_res[1:nlatlon,i] = as.vector(as.matrix(csv_res))
  data_tem[1:nlatlon,i] = as.vector(as.matrix(csv_tem))
  data_pre[1:nlatlon,i] = as.vector(as.matrix(csv_pre))
}

#################### function ##############
funVariationDecomp <- function(bas, cc, cp, cs,cue,gpp,npp,pre,res,tem){
  library(hier.part)
  library(gtools)
  library(rdaenvpart)
  
  cv_all =  matrix(nrow = 6, ncol = 1, NaN)

  xc_ln  = log(cc)
  npp_ln = log(npp)
  res_ln = log(res)
  gpp_ln = log(gpp)
  cue_ln = log(cue)
  bas_ln = log(bas)
  tem_ln = log(tem)
  pre_ln = log(pre)
  
  xc_ln[is.infinite(xc_ln)]   = NaN
  npp_ln[is.infinite(npp_ln)] = NaN
  res_ln[is.infinite(res_ln)] = NaN
  gpp_ln[is.infinite(gpp_ln)] = NaN
  cue_ln[is.infinite(cue_ln)] = NaN
  bas_ln[is.infinite(bas_ln)] = NaN
  tem_ln[is.infinite(tem_ln)] = NaN
  pre_ln[is.infinite(pre_ln)] = NaN
  
  tryCatch({
    # step 1: x : xc and xp
    xpc   = data.frame(cc,-cp)
    res1  = hier.part(cs,xpc,family = 'gaussian',gof = "Rsqu", barplot = FALSE)
    step1_xc_x = res1$I.perc[1,1]
    step1_xp_x = res1$I.perc[2,1]
    
    # step 2: xc : npp and tau 
    npp_res_ln = data.frame(npp_ln,res_ln)
    res2  = hier.part(xc_ln, npp_res_ln,family = 'gaussian',gof = "Rsqu", barplot = FALSE )
    step2_npp_xc = res2$I.perc[1,1]
    step2_res_xc = res2$I.perc[2,1]
    
    # step 3: npp : gpp and cue
    gpp_cue_ln = data.frame(gpp_ln,cue_ln)
    res3  = hier.part(npp_ln, gpp_cue_ln,family = 'gaussian',gof = "Rsqu", barplot = FALSE )
    step3_gpp_npp = res3$I.perc[1,1]
    step3_cue_npp = res3$I.perc[2,1]
    
    # step 4: tau--baseline Tau, temperature and precipitation
    bas_envs_ln = data.frame(bas_ln,-tem_ln, -pre_ln)
    res4  = hier.part(res_ln, bas_envs_ln,family = 'gaussian',gof = "Rsqu", barplot = FALSE )
    step4_bas_res = res4$I.perc[1,1]
    step4_tem_res = res4$I.perc[2,1]
    step4_pre_res = res4$I.perc[3,1]

    # total
    cv_p   = step1_xp_x
    cv_gpp = (((step3_gpp_npp/100)*step2_npp_xc)/100)*step1_xc_x
    cv_cue = (((step3_cue_npp/100)*step2_npp_xc)/100)*step1_xc_x
    cv_bas = (((step4_bas_res/100)*step2_res_xc)/100)*step1_xc_x
    cv_tem = (((step4_tem_res/100)*step2_res_xc)/100)*step1_xc_x
    cv_pre = (((step4_pre_res/100)*step2_res_xc)/100)*step1_xc_x

    cv_all[1] = cv_p
    cv_all[2] = cv_gpp
    cv_all[3] = cv_cue
    cv_all[4] = cv_bas
    cv_all[5] = cv_tem
    cv_all[6] = cv_pre

    res_value = which.max(cv_all)
    return(ifelse(length(res_value) ==0,NaN,res_value))
  }, error = function(e)NaN)
}

##################### Start to parallel run ##############
cores <- detectCores(logical=F)
cl    <- makeCluster(cores)
registerDoParallel(cl, cores=cores-1)

res_cv <- foreach(i=1:nlatlon,.combine='rbind') %dopar%
  {
    funVariationDecomp(data_bas[i,], data_cc[i,], data_cp[i,], data_cs[i,], data_cue[i,], data_gpp[i,],data_npp[i,],data_pre[i,],data_res[i,],data_tem[i,])
  }
stopCluster(cl)
res_cv = array(res_cv,dim=c(n_lat,n_lon))
write.table(res_cv, file = outFileName, row.names = F, col.names= F, sep=",")