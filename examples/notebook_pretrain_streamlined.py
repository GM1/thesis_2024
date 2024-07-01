# Databricks notebook source
import os
import sys
import argparse
import json
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import scanpy as sc
import numpy as np
import torch
import transformers
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
import datasets
from datasets import Dataset, load_dataset, concatenate_datasets


# sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.loss import masked_mse_loss, masked_relative_error
from scgpt.tokenizer import GeneVocab, random_mask_value
from scgpt.scbank import DataBank
from scgpt.utils import MainProcessOnly


# disable progress bar in datasets library
datasets.utils.logging.disable_progress_bar()

# COMMAND ----------

import sys
sys.version

# COMMAND ----------


