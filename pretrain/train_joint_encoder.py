import logging
import os
import argparse
import math
import sys
from typing import Iterable
import numpy as np
from pathlib import Path
import datetime
import json

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import timm
import timm.optim.optim_factory as optim_factory
import torch.nn as nn

import time

import wandb

from models import proteinbind_encoder_model
from models.proteinbind_encoder_model import ModalityType, load_module, save_module

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

from datasets.fasta_pdb_encoder_dataset import FastaPdbDataset
from datasets.fasta_text_encoder_dataset import FastaTextDataset
from dataloaders.joint_dataloader import JointDataLoader

import esm
from esm import inverse_folding


from einops import rearrange
from models.transformer import SimpleTransformer, BlockWithMasking, Mlp

from esm.inverse_folding.gvp_transformer import GVPTransformerModel
from esm.inverse_folding.gvp_transformer_encoder import GVPTransformerEncoder
from esm.inverse_folding.transformer_decoder import TransformerDecoder
from esm.inverse_folding.transformer_layer import TransformerEncoderLayer
from esm.modules import ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead, TransformerLayer

import faulthandler
faulthandler.enable()


def parse_args():
    parser = argparse.ArgumentParser(description="Train the ProteinBind model.")
    parser.add_argument("--seed", type=int, default=43, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training ('cpu' or 'cuda')")
    parser.add_argument("--datasets_dir", type=str, default="./datasets",
                        help="Directory containing the datasets")
    parser.add_argument("--datasets", type=str, nargs="+", default=["dreambooth"], choices=["dreambooth"],
                        help="Datasets to use for training and validation")
    parser.add_argument("--full_model_checkpoint_dir", type=str, default="./.checkpoints/full",
                        help="Directory to save the full model checkpoints")
    parser.add_argument("--full_model_checkpointing", action="store_true", help="Save full model checkpoints")

    parser.add_argument("--accum_iter", type=int, default=1, help="")
    parser.add_argument("--start_epoch", type=int, default=0, help="")
    parser.add_argument("--epochs", type=int, default=500, help="Maximum number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--momentum_betas", nargs=2, type=float, default=[0.9, 0.95],
                        help="Momentum beta 1 and 2 for Adam optimizer")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature parameter for InfoNCE loss")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--self_contrast", action="store_true", help="Use self-contrast on the image modality")
    parser.add_argument("--load_checkpoint_dir", type=str, default="./.checkpoints/lora",
                        help="Directory to save LoRA checkpoint")
    parser.add_argument("--lora_modality_names", nargs="+", type=str, default=["vision", "text"],
                        choices=["vision", "text", "audio", "thermal", "depth", "imu"],
                        help="Modality names to apply LoRA")
    parser.add_argument("--resume", type=str, default=None,
                        help="Directory to the saved checkpoint")

    parser.add_argument("--linear_probing", action="store_true",
                        help="Freeze model and train the last layers of the head for each modality.")

    parser.add_argument('--output_dir', default='./ckpt',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./ckpt',
                        help='path where to tensorboard log')

    # torch DDP setting
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', default=True, type=bool, help='')
    parser.add_argument('--world_size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # wandb settings
    parser.add_argument('--wandb_project', type=str, default='proteinbind')
    parser.add_argument('--wandb_run_name', type=str, default='seq-str-text')
    parser.add_argument('--wandb_mode', type=str, default='disabled')
    return parser.parse_args()


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('dual_nll', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('dual_nll_2', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    
    for data_iter_step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # batch 1 is sequence structure, batch 2 is sequence text
        batch1, batch2 = batch

        # data_a is sequence, data_b is structure
        data_a, class_a, data_b, class_b = batch1

        # data_a_2 is sequence, data_b_2 is text
        data_a_2, class_a_2, data_b_2, class_b_2 = batch2

        data_a = [data_a]
        data_b = [data_b]
        data_a_2 = [data_a_2]
        
        # TODO: class_a is always protein sequence (anchor modality)
        feats_a = [model({class_a: data_a_i.to(device=device)}) for data_a_i in data_a]
        feats_a_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_a], dim=0)
        # class_b could be any modality
        feats_b = [model({class_b: data_b_i}) for idx, data_b_i in enumerate(data_b)]
        feats_b_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_b], dim=0)


        # TODO: class_a is always protein sequence (anchor modality)
        feats_a_2 = [model({class_a_2: data_a_2_i.to(device=device)}) for data_a_2_i in data_a_2]
        feats_a_2_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_a_2], dim=0)
        # class_b could be any modality
        feats_b_2 = [model({class_b_2: data_b_2_i.to(device=device)}) for idx, data_b_2_i in enumerate(data_b_2)]
        feats_b_2_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_b_2], dim=0)
        
        if args.self_contrast:
            feats_a_b_tensor = torch.cat([feats_a_tensor.chunk(2)[0], feats_b_tensor], dim=0)
            feats_tensors = [feats_a_tensor, feats_a_b_tensor]
            temperatures = [1, args.temperature]
            contrast = ["self", "cross"]

            feats_a_b_2_tensor = torch.cat([feats_a_2_tensor.chunk(2)[0], feats_b_2_tensor], dim=0)
            feats_tensors_2 = [feats_a_2_tensor, feats_a_b_2_tensor]
        else:
            feats_a_b_tensor = torch.cat([feats_a_tensor, feats_b_tensor], dim=0)
            feats_tensors = [feats_a_b_tensor]
            temperatures = [args.temperature]
            contrast = ["cross"]

            feats_a_b_2_tensor = torch.cat([feats_a_2_tensor, feats_b_2_tensor], dim=0)
            feats_tensors_2 = [feats_a_b_2_tensor]

        # Accumulate self-contrastive loss for image and its augmentation, and modailty with image
        dual_nll = False
        for feats_idx, feats_tensor in enumerate(feats_tensors):
            # Calculate cosine similarity
            cos_sim = F.cosine_similarity(feats_tensor[:, None, :], feats_tensor[None, :, :], dim=-1)
            # Mask out cosine similarity to itself
            self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
            cos_sim.masked_fill_(self_mask, -float('inf'))
            # Find positive example -> batch_size//2 away from the original example
            pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
            # InfoNCE loss
            cos_sim = cos_sim / temperatures[feats_idx]
            nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
            nll = nll.mean()
            
            if not dual_nll:
                dual_nll = nll
            else:
                dual_nll += nll
                dual_nll /= 2

            # Get ranking position of positive example
            comb_sim = torch.cat(
                [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -float("inf"))],
                # First position positive example
                dim=-1,
            )
            sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        
        dual_nll_2 = False
        for feats_idx, feats_tensor in enumerate(feats_tensors_2):
            # Calculate cosine similarity
            cos_sim = F.cosine_similarity(feats_tensor[:, None, :], feats_tensor[None, :, :], dim=-1)
            # Mask out cosine similarity to itself
            self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
            cos_sim.masked_fill_(self_mask, -float('inf'))
            # Find positive example -> batch_size//2 away from the original example
            pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
            # InfoNCE loss
            cos_sim = cos_sim / temperatures[feats_idx]
            nll_2 = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
            nll_2 = nll_2.mean()
            
            if not dual_nll_2:
                dual_nll_2 = nll_2
            else:
                dual_nll_2 += nll_2
                dual_nll_2 /= 2
        
        loss_value = dual_nll.item() + dual_nll_2.item()


        loss_value_reduce = misc.all_reduce_mean(loss_value)

        dual_nll_reduce = misc.all_reduce_mean(dual_nll.item())
        dual_nll_2_reduce = misc.all_reduce_mean( dual_nll_2.item())
        dual_nll += dual_nll_2
        dual_nll /= accum_iter

        # Loss backward and optimizer step is all implemented inside scaler call
        loss_scaler(dual_nll, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value_reduce)
        metric_logger.update(dual_nll=dual_nll_reduce)
        metric_logger.update(dual_nll_2=dual_nll_2_reduce)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0 and misc.get_rank() == 0:
            log_writer.log({'train_loss': loss_value_reduce})
            log_writer.log({'lr': lr})
            log_writer.log({'dual_nll': dual_nll_reduce})
            log_writer.log({'dual_nll_2': dual_nll_2_reduce})


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def main(args):
    misc.init_distributed_mode(args)
    if misc.get_rank() == 0:
        print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    seq_encoder, seq_alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    str_encoder, str_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

    str_encoder = str_encoder.encoder


    # we do not use these in proteinbind, keepng these will cause DDP unused parameter error
    del seq_encoder.contact_head
    del seq_encoder.lm_head

    # initialize model
    model = proteinbind_encoder_model.proteinbind_huge(seq_encoder=seq_encoder, str_encoder=str_encoder).to(device)
    model_without_ddp = model

    # TODO: add dataloader and dataset in this place
    dataset_pdb_seq_train = FastaPdbDataset(
        pdb_alphabet=str_alphabet,
        seq_alphabet=seq_alphabet,
        mapping="PATH/TO/MAPPING",
        pdb_root="PATH/TO/PDB",
        fasta_root="PATH/TO/FASTA",
        split="train",
        train_size=0.99)
    
    
    dataset_txt_seq_train = FastaTextDataset(
        seq_alphabet=seq_alphabet,
        mapping="PATH/TO/MAPPING",
        fasta_root="PATH/TO/FASTA",
        split="train",
        train_size=0.99)

    effective_range = args.batch_size * misc.get_world_size() * args.accum_iter

    if misc.get_rank() == 0:
        print(effective_range)

    # initialize ddp training scheme
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_pdb_seq_train = torch.utils.data.DistributedSampler(
        dataset_pdb_seq_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )


    if misc.get_rank() == 0:
        print("Sampler_train = %s" % str(sampler_pdb_seq_train))

    sampler_txt_seq_train = torch.utils.data.DistributedSampler(
        dataset_txt_seq_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    if misc.get_rank() == 0:
        print("Sampler_train = %s" % str(sampler_txt_seq_train))

    if global_rank == 0 and args.log_dir is not None:
        # os.makedirs(args.log_dir, exist_ok=True)
        # log_writer = SummaryWriter(log_dir=args.log_dir)
        print("initialize wandb")
        # wandb.login()
        log_writer = wandb.init(project=args.wandb_project, name=args.wandb_run_name, mode=args.wandb_mode)

    else:
        log_writer = None

    data_loader_seq_pdb_train = torch.utils.data.DataLoader(
        dataset_pdb_seq_train, sampler=sampler_pdb_seq_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset_pdb_seq_train.collate_fn,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_seq_txt_train = torch.utils.data.DataLoader(
        dataset_txt_seq_train, sampler=sampler_txt_seq_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset_txt_seq_train.collate_fn,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    print("Model = %s" % str(model_without_ddp))
    # exit()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay,
                            betas=args.momentum_betas)
    if misc.get_rank() == 0:
        print(optimizer)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 50
    )
    # gradient scaler enabling fp16 training
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    print(args.start_epoch)

    if misc.get_rank() == 0:
        print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_seq_pdb_train.sampler.set_epoch(epoch)
            data_loader_seq_txt_train.sampler.set_epoch(epoch)
        
        joint_dataloader = JointDataLoader(data_loader_seq_pdb_train, data_loader_seq_txt_train)
        train_stats = train_one_epoch(
            model, joint_dataloader,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        lr_scheduler.step()

        if args.output_dir and (epoch % 1 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            '''
            if log_writer is not None:
                log_writer.flush()
            '''
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

