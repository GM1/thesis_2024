# Databricks notebook source
import gc
import math
from typing import Dict, Mapping, Optional, Tuple, Any, Union

import torch
import numpy as np
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Bernoulli
from tqdm import trange
import warnings

try:
    from flash_attn.modules.mha import MHA
    from flash_layers import FlashscGPTLayer, FlashscGPTGenerator
    flash_attn_available = True
except ImportError:
    warnings.warn("flash_attn is not installed")
    print("Error importing modules, flash attention is unavailable")
    flash_attn_available = False

from .dsbn import DomainSpecificBatchNorm1d
from .grad_reverse import grad_reverse


# COMMAND ----------

from flash_attn.modules.mha import MHA

# COMMAND ----------

from flash_layers import FlashscGPTLayer, FlashscGPTGenerator

# COMMAND ----------

from flash_attn import flash_attn_qkvpacked_func

# COMMAND ----------

from flash_attn import flash_attn_varlen_qkvpacked_func

# COMMAND ----------


