# Databricks notebook source
pip freeze > requirements.txt

# COMMAND ----------

os.getcwd()

# COMMAND ----------

# pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %md
# MAGIC # Code

# COMMAND ----------

pip install scanpy scib torchtext==0.15.2

# COMMAND ----------

pip list | grep torch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports

# COMMAND ----------

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
# from scgpt import logger

# COMMAND ----------

# disable progress bar in datasets library
datasets.utils.logging.disable_progress_bar()

# COMMAND ----------

# MAGIC %md
# MAGIC # Constants/Args

# COMMAND ----------

# TODO: convert to constants.py

class CustomObject:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)

arguments = {
    "data_source": "",      # Input datasource
    "save_dir": "examples/", # Where to save the model file
    "load_model": None, # Where to load the model from so that training can be restarted
    "n_hvg": None, # Number of highly variable genes If set to 0, will use all genes.
                   # Default is None, which will determine the n_hvg automatically.
    "valid_size_or_ratio": 0.03, # The ratio or size of the validation set size if split the dataset.
                                # If value is between 0 and 1, will be parsed as the ratio. If value is
                                # greater than 1 and be an integer, will be parsed as the size. If value 
                                # is 0, will not split the dataset.

    "grad_accu_steps": 1, # The number of gradient accumulation steps. Default is 1.
    "pad_token": "<pad>", # The token to use for padding. Default is <pad>.
    "input_style": "binned", # ["normed_raw", "log1p", "binned"] Default is binned.
    "input_emb_style": "continuous", # The style of the input embedding. Default is continuous. ["category", "continuous", "scaling"]
    "n_bins": 51, # The number of bins to use for the binned input style. Default is 51.
    "max_seq_len": 1536, # The maximum length of the sequence. Default is 1000. The actual used "
                         # "max length would be the minimum of this value and the length of the longest "
                         # "sequence in the data."

    "training_tasks": "both", # ["pcpt", "gen", "both"] "The tasks to use for training. pcpt: perception training with masked token "
                          # "learning. gen: generation. Default is both."

    "mask_ratio": 0.40, # The ratio of masked values in the training data. Default is 0.40. This"
                        # "value will be ignored if --training-tasks is set to gen or both.

    "trunc_by_sample": False, # Whether to truncate the input by sampling rather than cutting off if "
                       # "sequence length > max_seq_length. Default is False.

    "vocab_path": "", # Path to the vocabulary file.
    "local_rank": -1, # The local rank of the process for using the torch.distributed.launch "
                      # "utility. Will be -1 if not running in distributed model.

    "batch_size": 32, # The batch size for training. Default is 32.
    "eval_batch_size": 32, # The batch size for evaluation. Default is 32.
    "epochs": 15, # The number of epochs for training.
    "lr": 1e-3, # 1e-3, # The learning rate for training. Default is 1e-3.

    "scheduler_interval": 100, # The interval iterations for updating the learning rate. Default is 100. "
                               # "This will only be used when warmup-ratio is 0.

    "scheduler_factor": 0.99, # The factor for updating the learning rate. Default is 0.99. "
                              # "This will only be used when warmup-ratio is 0.

    "warmup_ratio_or_step": 0.1, # The ratio of warmup steps out of the total training steps. Default is 0.1. "
                                 # "If warmup-ratio is above 0, will use a cosine scheduler with warmup. If "
                                 # "the value is above 1, will use it as the number of warmup steps.

    "no_cls": False, # Whether to deactivate the classification loss. Default is False.
    "no_cce": True, # Whether to deactivate the contrastive cell embedding objective. "
                    # "Default is False.

    "fp16": False, # Whether to train in automatic mixed precision. Default is False.
    "fast_transformer": False, # Whether to use the fast transformer. Default is True.
    "nlayers": 4, # The number of layers for the transformer. Default is 4.
    "nheads": 4, # The number of layers for the transformer. Default is 4.
    "embsize": 64, # The embedding size for the transformer. Default is 64.
    "d_hid": 64, # Dimension of the feedforward network model in the transformer. Default is 64.
    "dropout": 0.2, # The dropout rate. Default is 0.2.
    "n_layers_cls": 3, # The number of layers for the classification network, including the output layer. Default is 3.
    "log_interval": 500, # The interval for logging. Default is 100.
    "save_interval": 1000, # The interval for saving the model. Default is 1000.

}

args = CustomObject(arguments)

args.data_source = "/Volumes/kvai_usr_gmahon1/thesis/test_data/heart"#"/Volumes/kvai_usr_gmahon1/cellxgene/scb_storage_dev/heart.scb"
args.save_dir = "/Volumes/kvai_usr_gmahon1/scgpt/model_checkpoints"
args.vocab_path = "/Volumes/kvai_usr_gmahon1/scgpt/vocab_files/vocab.json"

# COMMAND ----------

assert args.input_style in ["normed_raw", "log1p", "binned"]
assert args.input_emb_style in ["category", "continuous", "scaling"]
if args.input_style == "binned":
    if args.input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned _t.")
elif args.input_style == "log1p" or args.input_style == "normed_raw":
    if args.input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw _t."
        )

if args.input_emb_style == "category":
    args.mask_value = args.n_bins + 1
    args.pad_value = args.n_bins  # for padding gene expr values
    n_input_bins = args.n_bins + 2
else:
    args.mask_value = -1
    args.pad_value = -2
    n_input_bins = args.n_bins

if args.training_tasks in ["gen", "both"]:
    args.mask_ratio = 0.4 #[0.25, 0.50, 0.75]

# COMMAND ----------

# MAGIC %md
# MAGIC # CUDA/Device level

# COMMAND ----------

special_tokens = [args.pad_token, "<cls>", "<eoc>"]
USE_CLS = not args.no_cls
USE_CCE = not args.no_cce
MVC = True
USE_GENERATIVE_TRAINING = True if args.training_tasks in ["gen", "both"] else False

IS_DATA_PARALLEL = args.local_rank != -1
if IS_DATA_PARALLEL:
    # These two lines is to solve issue #1 based on the suggestion from
    # https://discuss.pytorch.org/t/94382
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.local_rank)

    torch.distributed.init_process_group(
        backend="nccl",
        rank=args.local_rank,
        timeout=timedelta(hours=10),
    )
    # specify device 0 since the CUDA_VISIBLE_DEVICES is set to one GPU
    # https://discuss.pytorch.org/t/67488/4
    device = torch.device("cuda:0")
    n_gpu = torch.cuda.device_count()
    world_size = torch.distributed.get_world_size()
    print(
        f"device: {device} in world size {world_size}, "
        f"visible gpu(s): {os.environ['CUDA_VISIBLE_DEVICES']}/{n_gpu}"
    )
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# COMMAND ----------

save_dir = Path(args.save_dir)
# if args.local_rank in [0, -1]:
#     # save_dir.mkdir(parents=True, exist_ok=True)
#     # with open(save_dir / "args.json", "w") as f:
#     #     json.dump(vars(args), f, indent=2)
#     # # copy all uncommitted changes to the save dir
#     # os.system(
#     #     f"git diff > {str(save_dir / 'git_diff_')}{scg.utils.get_git_commit()}.diff"
#     # )
if IS_DATA_PARALLEL:
    torch.distributed.barrier()

# scg.utils.add_file_handler(logger, save_dir / "run.log")
# log running date and current git commit
print(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
# print(f"Current git commit: {scg.utils.get_git_commit()}")

writer = SummaryWriter(log_dir=save_dir / "tensorboard")
if IS_DATA_PARALLEL:
    writer = MainProcessOnly(writer)

# COMMAND ----------

# MAGIC %md
# MAGIC # PREP & LOAD DATASETS

# COMMAND ----------

# TODO: Determine how exactly this function works and rebuild it for my own implementation

def _map_append_cls(dataset: Dataset) -> Dataset:
    print(f"Rank {args.local_rank}: Appending <cls> to dataset")
    dataset = dataset.map(
        lambda example: {
            "genes": [vocab["<cls>"]] + example["genes"],
            "expressions": [args.pad_value] + example["expressions"],
        },
        # batched=True,  # not using since then the map func needs to loop
        num_proc=len(os.sched_getaffinity(0)),
    )

    return dataset

# COMMAND ----------

with open("/Volumes/kvai_usr_gmahon1/thesis/test_data/cache/parquet/default-01d0ee3f4c006572/0.0.0/ca31c69184d9832faed373922c2acccec0b13a0bb5bbbe19371385c3ff26f1d1/parquet-train.arrow", "rb") as f:
    arrow = f.readlines()

# COMMAND ----------

arrow

# COMMAND ----------

d = spark.read.load("/Volumes/kvai_usr_gmahon1/thesis/test_data/cache/parquet/default-01d0ee3f4c006572/0.0.0/ca31c69184d9832faed373922c2acccec0b13a0bb5bbbe19371385c3ff26f1d1/")

# COMMAND ----------

if args.load_model is not None:
    model_dir = Path(args.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    if len(vocab) != len(json.load(open(model_dir / "vocab.json"))):
        raise ValueError(
            f"The vocabulary in the model directory to load ({model_dir}) does "
            "not match the current vocabulary. "
        )
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    if args.pad_token != model_configs["pad_token"]:
        print(
            f"The pad token in the model directory to load ({model_dir}) "
            "does not match the current pad token. Be careful if this is not expected."
        )
    if args.pad_value != model_configs["pad_value"]:
        print(
            f"The pad value in the model directory to load ({model_dir}) "
            "does not match the current pad value. Be careful if this is not expected."
        )
    print(
        f"Resume model from {model_file}, the model args will be overridden the "
        f"config {model_config_file}."
    )
    args.embsize = model_configs["embsize"]
    args.nheads = model_configs["nheads"]
    args.d_hid = model_configs["d_hid"]
    args.nlayers = model_configs["nlayers"]
    args.n_layers_cls = model_configs["n_layers_cls"]

    # resave the args with the new values
    if args.local_rank in [0, -1]:
        with open(save_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

# save the vocabulary
if args.local_rank in [0, -1]:
    with open(save_dir / "vocab.json", "w") as f:
        json.dump(
            {token: index for token, index in vocab.get_stoi().items()},
            f,
            indent=2,
        )
if IS_DATA_PARALLEL:
    torch.distributed.barrier()  # wait for saving all the files

# COMMAND ----------

# TODO: This is the only relevant method so far, extract it...
if Path(args.data_source).is_dir():
    # collection of parquet files
    files = [str(f) for f in Path(args.data_source).glob("*.parquet")]
    cache_location = Path(args.data_source).parent / "cache"
    # TODO: Change to loading from parquet file instead of json...
    vocab = GeneVocab.from_file(Path(args.vocab_path))
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    if USE_CCE or USE_CLS or MVC:
        # load or make the dataset w/ <cls> appended at the beginning
        cls_prefix_datatable = Path(args.data_source) / "cls_prefix_data.parquet"
        # Build the cache data...
        if not cls_prefix_datatable.exists():
            if args.local_rank in [0, -1]:
                print(f"Rank {args.local_rank}: Preparing dataset")
                # load dataset is a torch datasets function
                raw_dataset = load_dataset(
                    "parquet",
                    data_files=files,
                    split="train",
                    cache_dir=str(cache_location),
                )
                raw_dataset = _map_append_cls(raw_dataset)
                raw_dataset.to_parquet(str(cls_prefix_datatable))
            if IS_DATA_PARALLEL:
                torch.distributed.barrier()  # wait for the mapping to finish
        raw_dataset = load_dataset(
            "parquet",
            data_files=str(cls_prefix_datatable),
            split="train",
            cache_dir=str(cache_location),
        )
        print(f"Loaded {len(raw_dataset)} examples from {cls_prefix_datatable}")
else:
    raise Exception("Only support loading from a directory of parquet files")

# COMMAND ----------

# Reformat raw dataset file to torch...
raw_dataset = raw_dataset.with_format("torch")

# split train and validation sets
raw_dataset = raw_dataset.train_test_split(
    test_size=args.valid_size_or_ratio, shuffle=True
)
train_dataset = raw_dataset["train"]
valid_dataset = raw_dataset["test"]
print(f"train set number of samples: {len(train_dataset)}, ")
print(f"valid set number of samples: {len(valid_dataset)}, ")

# COMMAND ----------

print(args.mask_ratio), print(args.training_tasks)

# COMMAND ----------

# MAGIC %md
# MAGIC # DATA COLLATOR

# COMMAND ----------

# TODO: Need to rebuild this collator to be my own version of same, work with this for now...
# NOTE: Binning is handled during initial data processing in my version...
collator = scg.DataCollator(
    do_padding=True if args.max_seq_len is not None else False,
    pad_token_id=vocab[args.pad_token],
    pad_value=args.pad_value,
    do_mlm=True,
    do_binning=True if args.input_style == "binned" else False,
    mlm_probability=args.mask_ratio,
    mask_value=args.mask_value,
    max_length=args.max_seq_len,
    sampling=args.trunc_by_sample,
    data_style=args.training_tasks,
)

# TODO: try batch sampler, train_sampler = BatchSampler()
train_sampler = (
    DistributedSampler(train_dataset)
    if IS_DATA_PARALLEL
    else RandomSampler(train_dataset)
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=train_sampler,
    collate_fn=collator,
    drop_last=True,
    num_workers=min(len(os.sched_getaffinity(0)), args.batch_size),
    pin_memory=True,
    prefetch_factor=4,
)
valid_sampler = (
    DistributedSampler(valid_dataset, shuffle=False)
    if IS_DATA_PARALLEL
    else SequentialSampler(valid_dataset)
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=args.eval_batch_size,
    sampler=valid_sampler,
    collate_fn=collator,
    drop_last=True,
    num_workers=min(len(os.sched_getaffinity(0)), args.eval_batch_size),
    pin_memory=True,
)

# COMMAND ----------

raw_dataset

# COMMAND ----------

if USE_CLS:
    # celltypes_labels = raw_dataset["celltypes"]
    celltypes_labels = ["heart"]
    num_types = len(set(celltypes_labels))
    celltypes_labels = np.array(celltypes_labels)

# COMMAND ----------

args.input_emb_style

# COMMAND ----------

# TODO: Ultimately need to build my own version of this as well...
ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    d_model=args.embsize,
    nhead=args.nheads,
    d_hid=args.d_hid,
    nlayers=args.nlayers,
    nlayers_cls=args.n_layers_cls,
    n_cls=num_types if USE_CLS else 1,
    # vocab=vocab,
    dropout=args.dropout,
    pad_token=args.pad_token,
    pad_value=args.pad_value,
    do_mvc=MVC,
    do_dab=False,
    use_batch_labels=False,  # TODO: try using batch labels, may help MVC
    input_emb_style=args.input_emb_style,
    n_input_bins=n_input_bins,
    use_generative_training=USE_GENERATIVE_TRAINING,
    use_fast_transformer=args.fast_transformer,
    fast_transformer_backend="flash",
)
if args.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
    except:
        from collections import OrderedDict

        params = OrderedDict()
        for key, value in torch.load(model_file).items():
            params[key.replace("module.", "")] = value
        model.load_state_dict(params)
model.to(device)
print(model)
if IS_DATA_PARALLEL:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device,
        find_unused_parameters=False,
    )


criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# setup scheduler
if args.warmup_ratio_or_step > 0:
    total_num_batches = len(train_loader) * args.epochs
    warmup_steps = (
        int(total_num_batches * args.warmup_ratio_or_step)
        if args.warmup_ratio_or_step < 1
        else int(args.warmup_ratio_or_step)
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_num_batches,
        last_epoch=-1,
    )
else:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.scheduler_interval, gamma=args.scheduler_factor
    )

# amp fp16 training
scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)



# COMMAND ----------


def train(model: nn.Module, train_loader: DataLoader, epoch: int) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse, total_cls, total_gen, total_mvc = 0.0, 0.0, 0.0, 0.0, 0.0
    total_error = 0.0
    log_interval = args.log_interval
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, data_dict in enumerate(train_loader):
        global_iter = epoch * num_batches + batch

        data_dict = {k: v.to(device) for k, v in data_dict.items()}
        data_dict["celltypes"] = torch.tensor([0]*args.batch_size)

        if USE_GENERATIVE_TRAINING:
            pcpt_gene = data_dict["pcpt_gene"]
            pcpt_expr = data_dict["pcpt_expr"]
            pcpt_key_padding_mask = pcpt_gene.eq(vocab[args.pad_token])
            gen_gene = data_dict["gen_gene"]
            gen_expr_target = target_values = data_dict["gen_expr_target"]
            gen_key_padding_mask = gen_gene.eq(vocab[args.pad_token])
        else:
            input_gene_ids = data_dict["gene"]
            input_values = data_dict["masked_expr"]
            target_values = data_dict["expr"]
            src_key_padding_mask = input_gene_ids.eq(vocab[args.pad_token])

        with torch.cuda.amp.autocast(enabled=args.fp16):
            if USE_GENERATIVE_TRAINING:
                output_dict = model(
                    pcpt_gene,
                    pcpt_expr,
                    pcpt_key_padding_mask,
                    gen_gene,
                    gen_key_padding_mask,
                    CLS=USE_CLS,
                    MVC=MVC,
                    generative_training=True,
                )
                gen_expr_preds = output_values = output_dict["gen_preds"]

                positions_to_match = ~gen_key_padding_mask
                loss = loss_mse = criterion(
                    gen_expr_preds, gen_expr_target, positions_to_match
                )
                writer.add_scalar("train/mse", loss_mse, global_iter)
                if MVC:
                    loss_mvc = criterion(
                        output_dict["mvc_output"][:, pcpt_gene.shape[1] :],
                        gen_expr_target,
                        positions_to_match,
                    )
                    loss = loss + loss_mvc
                    writer.add_scalar("train/mvc", loss_mvc, global_iter)
            else:
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=USE_CLS,
                    CCE=USE_CCE,  # TODO: move these flags to model's attributes
                    MVC=MVC,
                    generative_training=False,
                )
                output_values = output_dict["mlm_output"]

                positions_to_match = input_values.eq(
                    args.mask_value
                )  # the postions to predict
                loss = loss_mse = criterion(
                    output_values, target_values, positions_to_match
                )
                writer.add_scalar("train/mse", loss_mse, global_iter)
                if USE_CLS:
                    target_labels = data_dict["celltypes"]
                    loss_cls = criterion_cls(output_dict["cls_output"], target_labels)
                    loss = loss + loss_cls
                    writer.add_scalar("train/cls", loss_cls, global_iter)
                if USE_CCE:
                    loss_cce = 10 * output_dict["loss_cce"]
                    loss = loss + loss_cce
                    writer.add_scalar("train/cce", loss_cce, global_iter)
                if MVC:
                    loss_mvc = criterion(
                        output_dict["mvc_output"], target_values, positions_to_match
                    )
                    loss = loss + loss_mvc
                    writer.add_scalar("train/mvc", loss_mvc, global_iter)
            writer.add_scalar("train/loss", loss, global_iter)

            if USE_GENERATIVE_TRAINING and global_iter > 1000:
                previous_cell_embs = output_dict["cell_emb"].detach()
                preds = model(
                    pcpt_gene,
                    pcpt_expr,
                    pcpt_key_padding_mask,
                    gen_gene,
                    gen_key_padding_mask,
                    CLS=False,
                    MVC=False,
                    input_cell_emb=previous_cell_embs,
                    generative_training=True,
                )["gen_preds"]
                loss_gen = criterion(preds, gen_expr_target, positions_to_match)
                loss = loss + loss_gen
                writer.add_scalar("train/gen", loss_gen, global_iter)

                # TODO: try this choice of using a separate backprop
                # # this part is for the choice of using a separate backprop
                # model.zero_grad()
                # scaler.scale(loss_gen).backward()
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(
                #     model.parameters(),
                #     1.0,
                #     error_if_nonfinite=False if scaler.is_enabled() else True,
                # )
                # scaler.step(optimizer)
                # scaler.update()

        if args.grad_accu_steps > 1:
            loss = loss / args.grad_accu_steps
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if args.grad_accu_steps > 1:
            if batch % args.grad_accu_steps == 0 or batch == num_batches - 1:
                scheduler.step()
                optimizer.zero_grad()
        else:
            scheduler.step()
            optimizer.zero_grad()

        with torch.no_grad():
            mre = masked_relative_error(
                output_values, target_values, positions_to_match
            )
            writer.add_scalar("train/mre", mre, global_iter)

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_cls += loss_cls.item() if USE_CLS else 0.0
        total_gen += loss_gen.item() if "loss_gen" in locals() else 0.0
        total_mvc += loss_mvc.item() if MVC else 0.0
        total_error += mre.item()
        if args.local_rank in [0, -1] and batch % log_interval == 0 and batch > 0:
            # Writer logs gradients distribution
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    writer.add_histogram(name + "_grad", param.grad, global_iter)
                    writer.add_histogram(name + "_param", param, global_iter)

            # Log scalar values
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_cls = total_cls / log_interval if USE_CLS else 0.0
            cur_gen = total_gen / log_interval if "loss_gen" in locals() else 0.0
            cur_mvc = total_mvc / log_interval if MVC else 0.0
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            print(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                + (f"cls {cur_cls:5.2f} | " if USE_CLS else "")
                + (f"gen {cur_gen:5.2f} |" if "loss_gen" in locals() else "")
                + (f"mvc {cur_mvc:5.2f} |" if MVC else "")
            )
            writer.add_scalar("lr", lr, global_iter)

            total_loss = 0
            total_mse = 0
            total_cls = 0
            total_gen = 0
            total_mvc = 0
            total_error = 0
            start_time = time.time()

        # immediately eval and save
        if batch % args.save_interval == 0 and batch > 0:
            eval_and_save(model, valid_loader, global_iter)
            model.train()  # important, reset to train mode


# COMMAND ----------

validation_set = list(valid_loader)

# COMMAND ----------

validation_set[0]['expr'][0].shape

# COMMAND ----------

validation_set[0]['gene'][0]

# COMMAND ----------

validation_set[0]['expr'][0].unique().shape

# COMMAND ----------

def evaluate(model: nn.Module, valid_loader: DataLoader) -> Dict[str, torch.Tensor]:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    with torch.no_grad():
        for data_dict in valid_loader:
            data_dict = {k: v.to(device) for k, v in data_dict.items()}
            if USE_GENERATIVE_TRAINING:
                pcpt_gene = data_dict["pcpt_gene"]
                pcpt_expr = data_dict["pcpt_expr"]
                pcpt_key_padding_mask = pcpt_gene.eq(vocab[args.pad_token])
                gen_gene = data_dict["gen_gene"]
                gen_expr_target = target_values = data_dict["gen_expr_target"]
                gen_key_padding_mask = gen_gene.eq(vocab[args.pad_token])
            else:
                input_gene_ids = data_dict["gene"]
                input_values = data_dict["masked_expr"]
                target_values = data_dict["expr"]
                src_key_padding_mask = input_gene_ids.eq(vocab[args.pad_token])

            with torch.cuda.amp.autocast(enabled=args.fp16):
                if USE_GENERATIVE_TRAINING:
                    output_dict = model(
                        pcpt_gene,
                        pcpt_expr,
                        pcpt_key_padding_mask,
                        gen_gene,
                        gen_key_padding_mask,
                        CLS=False,
                        MVC=False,
                        generative_training=True,
                    )
                    gen_expr_preds = output_values = output_dict["gen_preds"]

                    positions_to_match = ~gen_key_padding_mask
                else:
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        CLS=False,  # evaluation does not need CLS or CCE
                        CCE=False,
                        MVC=False,
                        generative_training=False,
                    )
                    output_values = output_dict["mlm_output"]
                    positions_to_match = input_values.eq(args.mask_value)

                loss = criterion(output_values, target_values, positions_to_match)
            total_loss += loss.item()
            total_error += masked_relative_error(
                output_values, target_values, positions_to_match
            ).item()
    total_loss = total_loss / len(valid_loader)
    total_error = total_error / len(valid_loader)
    return {
        "mse": torch.tensor(total_loss, device=device, dtype=torch.float),
        "mre": torch.tensor(total_error, device=device, dtype=torch.float),
    }


def eval_and_save(
    model: nn.Module,
    valid_loader: DataLoader,
    iter_or_epoch: int,
    is_epoch: bool = False,
    save: bool = True,
) -> None:
    # perform evaluation in distributed data parallel
    val_loss, val_mre = evaluate(model, valid_loader).values()
    if IS_DATA_PARALLEL:
        # gather the results from all the processes
        val_loss_list = [torch.zeros_like(val_loss) for _ in range(world_size)]
        val_mre_list = [torch.zeros_like(val_mre) for _ in range(world_size)]
        torch.distributed.all_gather(val_loss_list, val_loss)
        torch.distributed.all_gather(val_mre_list, val_mre)
        val_loss = torch.mean(torch.stack(val_loss_list))
        val_mre = torch.mean(torch.stack(val_mre_list))
    val_loss, val_mre = val_loss.item(), val_mre.item()
    if args.local_rank in [0, -1]:
        if is_epoch:
            elapsed = time.time() - epoch_start_time
            print("-" * 89)
            print(
                f"| end of epoch {iter_or_epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
            )
            print(f"{'-' * 89}\n")
            writer.add_scalar("valid/mse", val_loss, iter_or_epoch * len(valid_loader))
            writer.add_scalar("valid/mre", val_mre, iter_or_epoch * len(valid_loader))
        else:
            print(f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}")
            writer.add_scalar("valid/mse", val_loss, iter_or_epoch)
            writer.add_scalar("valid/mre", val_mre, iter_or_epoch)

        global best_val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save the best model
            print(f"Saving the best model to {args.save_dir}")
            torch.save(
                model.module.state_dict()
                if isinstance(
                    model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
                )
                else model.state_dict(),
                args.save_dir + "/best_model.pt",
            )

        if save:
            torch.save(
                model.module.state_dict()
                if isinstance(
                    model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
                )
                else model.state_dict(),
                args.save_dir + f"/model-{'ep' if is_epoch else ''}{iter_or_epoch}.pt",
            )
    if IS_DATA_PARALLEL:
        torch.distributed.barrier()


# COMMAND ----------

for batch, data_dict in enumerate(train_loader):
    data_dict = {k: v.to(device) for k, v in data_dict.items()}
    data_dict["celltypes"] = torch.tensor([1]*args.batch_size)
    print(data_dict)
    break

# COMMAND ----------

# %%
best_val_loss = float("inf")
print("Start training")
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train(model, train_loader, epoch=epoch)
    eval_and_save(model, valid_loader, iter_or_epoch=epoch, is_epoch=True)

writer.flush()
writer.close()

# %%
# compare with the naive baseline of all ones
data_dict = next(iter(valid_loader))
input_values = data_dict["masked_expr"]
tagert_values = data_dict["expr"]
predict_ones = torch.ones(input_values.shape, dtype=torch.float32)
mse = masked_mse_loss(predict_ones, tagert_values, input_values.eq(args.mask_value))
mre = masked_relative_error(
    predict_ones, tagert_values, input_values.eq(args.mask_value)
)
print(f"MSE: {mse.item()}, MRE: {mre.item()}")

# %% [markdown]
# # Analysis
model.to(device)
model.eval()

# COMMAND ----------

# model_path = "/Volumes/kvai_usr_gmahon1/scgpt/model_checkpoints/best_model.pt"
# my_model = torch.load(model_path)

# COMMAND ----------

# my_model

# COMMAND ----------

# type(model)

# COMMAND ----------

# print("hello world")

# COMMAND ----------

# torch.save(model.module.state_dict()
# if isinstance(
#     model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
# )
# else model.state_dict(),
# args.save_dir + f"/model-20240521-A100-PCPT-AM.pt")

# COMMAND ----------

import shutil
shutil.copytree("/Volumes/kvai_usr_gmahon1/thesis/test_data", "/Volumes/kvai_usr_gmahon1/scgpt/thesis_data/test_data")

# COMMAND ----------

# if args.data_source.endswith("human"):
#     TISSUE_LIST = [
#         "heart",
#         "blood",
#         "brain",
#         "lung",
#         "kidney",
#         "intestine",
#         "pancreas",
#         "others",
#     ]
#     root_data_source = Path(args.data_source).parent
#     raw_dataset_list = []
#     vocab = GeneVocab.from_file(Path(args.vocab_path))
#     for tissue in TISSUE_LIST:
#         tissue_data_path = root_data_source / tissue
#         cls_prefix_datatable = (
#             tissue_data_path / "all_counts" / "cls_prefix_data.parquet"
#         )
#         cache_dir = tissue_data_path / "cache"
#         tissue_dataset = load_dataset(
#             "parquet",
#             data_files=str(cls_prefix_datatable),
#             split="train",
#             cache_dir=str(cache_dir),
#         )
#         print(f"Loaded {tissue} examples from {cls_prefix_datatable}")
#         raw_dataset_list.append(tissue_dataset)
#     print("merging dataset...")
#     raw_dataset = concatenate_datasets(raw_dataset_list)
#     print("done merging dataset")
#     for s in special_tokens:
#         if s not in vocab:
#             vocab.append_token(s)
            
# elif Path(args.data_source).is_dir() and args.data_source.endswith(".scb"):
#     # the large-scale data structure
#     db = DataBank.from_path(args.data_source)
#     raw_dataset = db.main_data.data
#     vocab: GeneVocab = db.gene_vocab
#     for s in special_tokens:
#         if s not in vocab:
#             vocab.append_token(s)
#     if USE_CCE or USE_CLS or MVC:
#         # load or make the dataset w/ <cls> appended at the beginning
#         cls_prefix_datatable = Path(args.data_source) / "cls_prefix_data.parquet"
#         if not cls_prefix_datatable.exists():
#             if args.local_rank in [0, -1]:
#                 raw_dataset = _map_append_cls(raw_dataset)
#                 raw_dataset.to_parquet(cls_prefix_datatable)
#             if IS_DATA_PARALLEL:
#                 torch.distributed.barrier()  # wait for the mapping to finish
#         raw_dataset = load_dataset(
#             "parquet",
#             data_files=str(cls_prefix_datatable),
#             split="train",
#             cache_dir=args.data_source,
#         )
#         print(f"Loaded {len(raw_dataset)} examples from {cls_prefix_datatable}")

# # TODO: This is the only relevant method so far, extract it...
# elif Path(args.data_source).is_dir():
#     # collection of parquet files
#     parquet_files = [str(f) for f in Path(args.data_source).glob("*.parquet")]
#     cache_dir = Path(args.data_source).parent / "cache"
#     vocab = GeneVocab.from_file(Path(args.vocab_path))
#     for s in special_tokens:
#         if s not in vocab:
#             vocab.append_token(s)
#     if USE_CCE or USE_CLS or MVC:
#         # load or make the dataset w/ <cls> appended at the beginning
#         cls_prefix_datatable = Path(args.data_source) / "cls_prefix_data.parquet"
#         # Build the cache data...
#         if not cls_prefix_datatable.exists():
#             if args.local_rank in [0, -1]:
#                 print(f"Rank {args.local_rank}: Preparing dataset")
#                 # load dataset is a torch datasets function
#                 raw_dataset = load_dataset(
#                     "parquet",
#                     data_files=parquet_files,
#                     split="train",
#                     cache_dir=str(cache_dir),
#                 )
#                 raw_dataset = _map_append_cls(raw_dataset)
#                 raw_dataset.to_parquet(str(cls_prefix_datatable))
#             if IS_DATA_PARALLEL:
#                 torch.distributed.barrier()  # wait for the mapping to finish
#         raw_dataset = load_dataset(
#             "parquet",
#             data_files=str(cls_prefix_datatable),
#             split="train",
#             cache_dir=str(cache_dir),
#         )
#         print(f"Loaded {len(raw_dataset)} examples from {cls_prefix_datatable}")

# elif Path(args.data_source).is_file():
#     adata = sc.read(args.data_source, cache=True)
#     # Specific the required column names, when loading the data the first time.
#     # Store the column names for later use.
#     (
#         celltype_col,
#         str_celltype_col,
#         gene_col,
#         batch_key,
#     ) = scg.utils.find_required_colums(
#         adata,
#         id=args.data_source,
#         configs_dir=Path(args.data_source).parent,
#     )
#     if celltype_col is None:
#         celltype_col = "int" + str_celltype_col
#         adata.obs[celltype_col] = scg.utils.category_str2int(
#             adata.obs[str_celltype_col]
#         )
        
# elif args.data_source == "test":  # Using test data
#     raw_dataset = Dataset.from_dict(
#         {
#             "id": [1] * 300,
#             "genes": [[1, 2, 3]] * 300,
#             "expressions": [[1.0, 2.0, 3.0]] * 300,
#         }
#     )
#     vocab = GeneVocab.from_dict({"zero": 0, "a": 1, "b": 2, "c": 3})
#     for s in special_tokens:
#         if s not in vocab:
#             vocab.append_token(s)
