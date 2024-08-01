# Databricks notebook source
# MAGIC %r
# MAGIC # Generate Simulated data for scRNA-seq denoising
# MAGIC # USING R 4.4.1
# MAGIC
# MAGIC # Step 1: Install the BiocManager
# MAGIC if (!requireNamespace("BiocManager", quietly = TRUE))
# MAGIC     install.packages("BiocManager")
# MAGIC
# MAGIC # Step 2: Install the required libraries using BiocManager
# MAGIC BiocManager::install("splatter")
# MAGIC BiocManager::install("zellkonverter")
# MAGIC BiocManager::install("Seurat")
# MAGIC BiocManager::install("patchwork")
# MAGIC BiocManager::install("scater")

# COMMAND ----------

# MAGIC %r
# MAGIC library(splatter)
# MAGIC library(scater)
# MAGIC library(patchwork)
# MAGIC library(DESeq2)
# MAGIC library(zellkonverter)

# COMMAND ----------

# MAGIC %r
# MAGIC ##group= 2 no dropout
# MAGIC filepath <- "C:\\Users\\gmaho\\Desktop\\ai_masters_thesis\\simulated_datasets\\sim_g2_no_dropout.h5ad"
# MAGIC nGroups=2
# MAGIC group.prob <- rep(1, nGroups) / nGroups
# MAGIC sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",seed=42) 
# MAGIC counts<-counts(sim)
# MAGIC libsizes <- colSums(counts)
# MAGIC size.factors <- libsizes/mean(libsizes)
# MAGIC logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
# MAGIC sim <- runUMAP(sim)
# MAGIC plotUMAP(sim, colour_by = "Group")
# MAGIC zellkonverter::writeH5AD(sim, X_name="counts", filepath)
# MAGIC
# MAGIC
# MAGIC ##group= 2 dropout=1
# MAGIC filepath <- "C:\\Users\\gmaho\\Desktop\\ai_masters_thesis\\simulated_datasets\\sim_g2_dropout_5.h5ad"
# MAGIC nGroups=2
# MAGIC dropout=5
# MAGIC group.prob <- rep(1, nGroups) / nGroups
# MAGIC sim <- splatSimulate(group.prob=group.prob,
# MAGIC                      nGenes=200,batchCells=2000,method = "groups",dropout.type='experiment',seed=42, dropout.shape=-1, dropout.mid=5) 
# MAGIC counts<-counts(sim)
# MAGIC libsizes <- colSums(counts)
# MAGIC size.factors <- libsizes/mean(libsizes)
# MAGIC logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
# MAGIC sim <- runUMAP(sim)
# MAGIC plotUMAP(sim, colour_by = "Group")
# MAGIC zellkonverter::writeH5AD(sim, X_name="counts", filepath)
# MAGIC
# MAGIC ##group= 4 no dropout
# MAGIC filepath <- "C:\\Users\\gmaho\\Desktop\\ai_masters_thesis\\simulated_datasets\\sim_g4_no_dropout.h5ad"
# MAGIC nGroups=4
# MAGIC group.prob <- rep(1, nGroups) / nGroups
# MAGIC sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",seed=42) 
# MAGIC counts<-counts(sim)
# MAGIC libsizes <- colSums(counts)
# MAGIC size.factors <- libsizes/mean(libsizes)
# MAGIC logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
# MAGIC sim <- runUMAP(sim)
# MAGIC plotUMAP(sim, colour_by = "Group")
# MAGIC zellkonverter::writeH5AD(sim, X_name="counts", filepath)
# MAGIC
# MAGIC ##group= 4 dropout=1
# MAGIC filepath <- "C:\\Users\\gmaho\\Desktop\\ai_masters_thesis\\simulated_datasets\\sim_g4_dropout_1.h5ad"
# MAGIC nGroups=4
# MAGIC group.prob <- rep(1, nGroups) / nGroups
# MAGIC sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",dropout.type='experiment',seed=42, dropout.shape=-1, dropout.mid=5) 
# MAGIC counts<-counts(sim)
# MAGIC libsizes <- colSums(counts)
# MAGIC size.factors <- libsizes/mean(libsizes)
# MAGIC logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
# MAGIC sim <- runUMAP(sim)
# MAGIC plotUMAP(sim, colour_by = "Group")
# MAGIC zellkonverter::writeH5AD(sim, X_name="counts", filepath)
# MAGIC
# MAGIC ##group= 6 no dropout
# MAGIC filepath <- "C:\\Users\\gmaho\\Desktop\\ai_masters_thesis\\simulated_datasets\\sim_g6_no_dropout.h5ad"
# MAGIC nGroups=6
# MAGIC group.prob <- rep(1, nGroups) / nGroups
# MAGIC sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",seed=42) 
# MAGIC counts<-counts(sim)
# MAGIC libsizes <- colSums(counts)
# MAGIC size.factors <- libsizes/mean(libsizes)
# MAGIC logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
# MAGIC sim <- runUMAP(sim)
# MAGIC plotUMAP(sim, colour_by = "Group")
# MAGIC zellkonverter::writeH5AD(sim, X_name="counts", filepath)
# MAGIC
# MAGIC ##group= 6 dropout=1
# MAGIC filepath <- "C:\\Users\\gmaho\\Desktop\\ai_masters_thesis\\simulated_datasets\\sim_g6_dropout_1.h5ad"
# MAGIC nGroups=6
# MAGIC group.prob <- rep(1, nGroups) / nGroups
# MAGIC sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",dropout.type='experiment',seed=42, dropout.shape=-1, dropout.mid=5) 
# MAGIC counts<-counts(sim)
# MAGIC libsizes <- colSums(counts)
# MAGIC size.factors <- libsizes/mean(libsizes)
# MAGIC logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
# MAGIC sim <- runUMAP(sim)
# MAGIC plotUMAP(sim, colour_by = "Group")
# MAGIC sim <- runUMAP(sim)
# MAGIC zellkonverter::writeH5AD(sim, X_name="counts", filepath)
# MAGIC
# MAGIC ##group= 8 no dropout
# MAGIC filepath <- "C:\\Users\\gmaho\\Desktop\\ai_masters_thesis\\simulated_datasets\\sim_g8_no_dropout.h5ad"
# MAGIC nGroups=8
# MAGIC group.prob <- rep(1, nGroups) / nGroups
# MAGIC sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",seed=42) 
# MAGIC counts<-counts(sim)
# MAGIC libsizes <- colSums(counts)
# MAGIC size.factors <- libsizes/mean(libsizes)
# MAGIC logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
# MAGIC sim <- runUMAP(sim)
# MAGIC plotUMAP(sim, colour_by = "Group")
# MAGIC zellkonverter::writeH5AD(sim, X_name="counts", filepath)
# MAGIC
# MAGIC ##group= 8 dropout=1
# MAGIC filepath <- "C:\\Users\\gmaho\\Desktop\\ai_masters_thesis\\simulated_datasets\\sim_g8_dropout_1.h5ad"
# MAGIC nGroups=8
# MAGIC group.prob <- rep(1, nGroups) / nGroups
# MAGIC sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",dropout.type='experiment',seed=42, dropout.shape=-1, dropout.mid=5) 
# MAGIC counts<-counts(sim)
# MAGIC libsizes <- colSums(counts)
# MAGIC size.factors <- libsizes/mean(libsizes)
# MAGIC logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
# MAGIC sim <- runUMAP(sim)
# MAGIC plotUMAP(sim, colour_by = "Group")
# MAGIC zellkonverter::writeH5AD(sim, X_name="counts", filepath)

# COMMAND ----------


