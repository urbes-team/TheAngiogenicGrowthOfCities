# Script for creating population clusters, using code for the City Clustering
# Algorithm (Rozenfeld et al. 2011)
library(osc)
library(ggplot2)
library(hrbrthemes)

setwd(".")

#####################################################
# GREATER LONDON OBSERVED RESULTS
years = seq(from=1831, to=2012, by=10)
rho_lim <- 1000
for (yr in years) {
  print(yr)
  rhotxtfile = sprintf("./Data/London/rhoyr_txt/rho%s.txt",
                       yr)
  rhoobs <- read.delim(rhotxtfile, header=TRUE, sep=',', )
  rhocut <- rhoobs[rhoobs$rho > rho_lim,]
  rhocca <- cca(rhocut, s=1)
  ggplot(rhocca$cluster, aes(x=long, y=lat, fill=cluster_id)) + geom_tile()
  gl <- rhocca$cluster[rhocca$cluster$cluster_id == 
                         modal(rhocca$cluster$cluster_id),]
  ggplot(gl, aes(x=long, y=lat, fill=cluster_id)) + geom_tile()
  clusterfile = sprintf("./Data/London/rhoyr_clusters/gl%s_%s.csv", yr,
                        rho_lim)
  write.csv(gl, file=clusterfile)
}
#####################################################
# GREATER LONDON SIMULATED RESULTS
years = seq(from = 1831, to = 2012, by =10)
rho_lim <- 1000
run_name <- "domain_tI_all_optimum10"
for (yr in years) {
  print(yr)
  rhotxtfile = sprintf("./Results/LdnPopulationGrowth/%s/rho_txt/rho%s.txt",
                       run_name, yr)
  rhoobs <- read.delim(rhotxtfile, header=TRUE, sep=',', )
  rhocut <- rhoobs[rhoobs$rho > rho_lim,]
  rhocca <- cca(rhocut, s=1)
  ggplot(rhocca$cluster, aes(x=long, y=lat, fill=cluster_id)) + geom_tile()
  gl <- rhocca$cluster[rhocca$cluster$cluster_id == 
                         modal(rhocca$cluster$cluster_id),]
  ggplot(gl, aes(x=long, y=lat, fill=cluster_id)) + geom_tile()
  clusterfile = 
    sprintf("./Results/LdnPopulationGrowth/%s/rho_clusters/rho%s_%s.txt",
            run_name, yr, rho_lim)
  write.csv(gl, file=clusterfile)
}
#####################################################
# SYDNEY OBSERVED RESULTS
years = seq(from=1851, to=2012, by=10)
rho_lim <- 300
for (yr in years) {
  print(yr)
  rhotxtfile = sprintf("./Data/Sydney/rhoyr_txt/rhosydneymet%s.txt", yr)
  rhoobs <- read.delim(rhotxtfile, header=TRUE, sep=',', )
  rhocut <- rhoobs[rhoobs$rho > rho_lim,]
  rhocca <- cca(rhocut, s=1)
  ggplot(rhocca$cluster, aes(x=long, y=lat, fill=cluster_id)) + geom_tile()
  gl <- rhocca$cluster[rhocca$cluster$cluster_id == 
                         modal(rhocca$cluster$cluster_id),]
  ggplot(gl, aes(x=long, y=lat, fill=cluster_id)) + geom_tile()
  clusterfile = sprintf("./Data/Sydney/rhoyr_clusters/ms%s_%s.csv", yr,
                        rho_lim)
  write.csv(gl, file=clusterfile)
}
#####################################################
# SYDNEY SIMULATED RESULTS
years = seq(from = 1851, to = 2012, by =10)
rho_lim <- 300
run_name <- "./Results/SydPopulationGrowth/optimum_params1881_D1_kappa3"
for (yr in years) {
  print(yr)
  rhotxtfile = sprintf("./%s/rho_txt/rho%s.txt", run_name, yr)
  rhoobs <- read.delim(rhotxtfile, header=TRUE, sep=',', )
  rhocut <- rhoobs[rhoobs$rho > rho_lim,]
  rhocca <- cca(rhocut, s=1)
  ggplot(rhocca$cluster, aes(x=long, y=lat, fill=cluster_id)) + geom_tile()
  gl <- rhocca$cluster[rhocca$cluster$cluster_id == 
                         modal(rhocca$cluster$cluster_id),]
  ggplot(gl, aes(x=long, y=lat, fill=cluster_id)) + geom_tile()
  clusterfile = 
    sprintf("./%s/rho_clusters/rho%s_%s.txt", 
            run_name, yr, rho_lim)
  write.csv(gl, file=clusterfile)
}

#####################################################