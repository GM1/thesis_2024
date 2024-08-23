# Generate Simulated data for scRNA-seq denoising
# USING R 4.4.1

library(splatter)
library(scater)
library(Seurat)
library(patchwork)
library(DESeq2)
library(zellkonverter)



##group= 2 no dropout
filepath <- "C:\\Users\\gmaho\\Desktop\\ai_masters_thesis\\simulated_datasets\\sim_g2_no_dropout.h5ad"
nGroups=2
group.prob <- rep(1, nGroups) / nGroups
sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",seed=42)
counts<-counts(sim)
libsizes <- colSums(counts)
size.factors <- libsizes/mean(libsizes)
logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
sim <- runUMAP(sim)
plotUMAP(sim, colour_by = "Group")
zellkonverter::writeH5AD(sim, X_name="counts", filepath)


##group= 2 dropout=1
filepath <- "C:\\Users\\gmaho\\Desktop\\ai_masters_thesis\\simulated_datasets\\sim_g2_dropout_5.h5ad"
nGroups=2
dropout=5
group.prob <- rep(1, nGroups) / nGroups
sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",dropout.type='experiment',seed=42, dropout.shape=-1, dropout.mid=5)
counts<-counts(sim)
libsizes <- colSums(counts)
size.factors <- libsizes/mean(libsizes)
logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
sim <- runUMAP(sim)
plotUMAP(sim, colour_by = "Group")
zellkonverter::writeH5AD(sim, X_name="counts", filepath)

##group= 4 no dropout
filepath <- "C:\\Users\\gmaho\\Desktop\\ai_masters_thesis\\simulated_datasets\\sim_g4_no_dropout.h5ad"
nGroups=4
group.prob <- rep(1, nGroups) / nGroups
sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",seed=42)
counts<-counts(sim)
libsizes <- colSums(counts)
size.factors <- libsizes/mean(libsizes)
logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
sim <- runUMAP(sim)
plotUMAP(sim, colour_by = "Group")
zellkonverter::writeH5AD(sim, X_name="counts", filepath)

##group= 4 dropout=1
filepath <- "C:\\Users\\gmaho\\Desktop\\ai_masters_thesis\\simulated_datasets\\sim_g4_dropout_1.h5ad"
nGroups=4
group.prob <- rep(1, nGroups) / nGroups
sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",dropout.type='experiment',seed=42, dropout.shape=-1, dropout.mid=5)
counts<-counts(sim)
libsizes <- colSums(counts)
size.factors <- libsizes/mean(libsizes)
logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
sim <- runUMAP(sim)
plotUMAP(sim, colour_by = "Group")
zellkonverter::writeH5AD(sim, X_name="counts", filepath)

##group= 6 no dropout
filepath <- "C:\\Users\\gmaho\\Desktop\\ai_masters_thesis\\simulated_datasets\\sim_g6_no_dropout.h5ad"
nGroups=6
group.prob <- rep(1, nGroups) / nGroups
sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",seed=42)
counts<-counts(sim)
libsizes <- colSums(counts)
size.factors <- libsizes/mean(libsizes)
logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
sim <- runUMAP(sim)
plotUMAP(sim, colour_by = "Group")
zellkonverter::writeH5AD(sim, X_name="counts", filepath)

##group= 6 dropout=1
filepath <- "C:\\Users\\gmaho\\Desktop\\ai_masters_thesis\\simulated_datasets\\sim_g6_dropout_3.h5ad"
nGroups=6
group.prob <- rep(1, nGroups) / nGroups
sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",dropout.type='experiment',seed=42, dropout.shape=-1, dropout.mid=3)
counts<-counts(sim)
libsizes <- colSums(counts)
size.factors <- libsizes/mean(libsizes)
logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
sim <- runUMAP(sim)
plotUMAP(sim, colour_by = "Group")
sim <- runUMAP(sim)
zellkonverter::writeH5AD(sim, X_name="counts", filepath)

##group= 8 no dropout
filepath <- "C:\\Users\\gmaho\\Desktop\\ai_masters_thesis\\simulated_datasets\\sim_g8_no_dropout.h5ad"
nGroups=8
group.prob <- rep(1, nGroups) / nGroups
sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",seed=42)
counts<-counts(sim)
libsizes <- colSums(counts)
size.factors <- libsizes/mean(libsizes)
logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
sim <- runUMAP(sim)
plotUMAP(sim, colour_by = "Group")
zellkonverter::writeH5AD(sim, X_name="counts", filepath)

##group= 8 dropout=1
filepath <- "C:\\Users\\gmaho\\Desktop\\ai_masters_thesis\\simulated_datasets\\sim_g8_dropout_1.h5ad"
nGroups=8
group.prob <- rep(1, nGroups) / nGroups
sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",dropout.type='experiment',seed=42, dropout.shape=-1, dropout.mid=5)
counts<-counts(sim)
libsizes <- colSums(counts)
size.factors <- libsizes/mean(libsizes)
logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
sim <- runUMAP(sim)
plotUMAP(sim, colour_by = "Group")
zellkonverter::writeH5AD(sim, X_name="counts", filepath)



nGroups=2
group.prob <- rep(1, nGroups) / nGroups
sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",seed=42)
sim <- logNormCounts(sim)
sim <- runUMAP(sim)
plotUMAP(sim, colour_by = "Group")


params <- newSplatParams()
params <- setParam(params, "dropout.type", "experiment")
params <- setParam(params, "dropout.mid", 0.5) # Example of adjusting a midpoint parameter
params <- setParam(params, "dropout.shape", -1) # Adjusting the shape parameter to increase dropout


nGroups=2
group.prob <- rep(1, nGroups) / nGroups
sim <- splatSimulate(group.prob=group.prob,
nGenes=200,batchCells=2000,
method = "groups",
dropout.type='experiment',
seed=42,
dropout.shape=-1,
dropout.mid=5)
sim <- logNormCounts(sim)
sim <- runUMAP(sim)
plotUMAP(sim, colour_by = "Group")


nGroups=2
group.prob <- rep(1, nGroups) / nGroups
sim <- splatSimulate(group.prob=group.prob,
nGenes=200,batchCells=2000,
method = "groups",
dropout.type='experiment',
seed=42,
dropout.shape=-1,
dropout.mid=5)
counts<-counts(sim)
libsizes <- colSums(counts)
size.factors <- libsizes/mean(libsizes)
logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
sim <- runUMAP(sim)
plotUMAP(sim, colour_by = "Group")
