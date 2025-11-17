from argparse import ArgumentParser
import os
from typing import List

import numpy as np
import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group
from torch.utils.data import DistributedSampler
from yacs.config import CfgNode

from models.unet import ResidualUNet
from models.gnn import GNN
from utils.loss import ChamferLossEF, LaplacianLossEF
from utils.mesh_utils_torch import random_laplacian_smoothing_torch, get_laplacian_tensor, compute_vertex_normals
from dataset import MeshDatasetTMS
from config.config import get_default_args


def collate_fn(batch: List[np.ndarray]):
    tensors = torch.stack([torch.from_numpy(batch) for batch in batch])
    return tensors


def _helper(batch: List[dict]):
    results = {}
    for key in batch[0].keys():
        results[key] = []
        for b in batch:
            results[key].append(b[key])
    return results


def main(config: CfgNode, args: dict):
    torch.backends.cudnn.benchmark = True

    tms_model = ResidualUNet(
        num_classes=1, num_input_channels=1,
        down_channels=config.model.encoder_channels,
        up_channels=config.model.decoder_channels,
        deep_supervision=False,
        input_shape=config.model.input_img_size,
        voxel_decoder=True,
        skips_only=True
    )

    gnn_model = GNN(
        channels=config.model.graph_channels,
        unet_channels=config.model.encoder_channels + config.model.decoder_channels,
        aggregate_indices=config.model.agg_indices,
        residual_block_num=config.model.residual_block_num,
        graph_conv_block_num=config.model.graph_conv_block_num,
        voxel_shape=config.model.input_img_size
    )

    try:
        is_ddp = (int(os.environ['WORLD_SIZE']) > 1)
    except KeyError:
        is_ddp = False

    if is_ddp:
        init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f'cuda:{local_rank}')
        tms_model = DistributedDataParallel(tms_model.to(device), device_ids=[local_rank], output_device=local_rank)
        gnn_model = DistributedDataParallel(gnn_model.to(device), device_ids=[local_rank], output_device=local_rank)
    else:
        local_rank = 0
        device = torch.device(f'cuda:{local_rank}')
        tms_model = tms_model.to(device)
        gnn_model = gnn_model.to(device)

    loss_chamfer = ChamferLossEF()
    loss_lap = LaplacianLossEF(lap_kind='trimesh')

    param_group = [
        {'params': tms_model.parameters(), 'lr': config.optimizer.lr_unet},
        {'params': gnn_model.parameters(), 'lr': config.optimizer.lr},
    ]
    optimizer = Adam(param_group)

    dataset = MeshDatasetTMS()
    if is_ddp:
        sampler = DistributedSampler(dataset, num_replicas=int(os.environ['WORLD_SIZE']),
                                     rank=local_rank, shuffle=True, drop_last=False)
        data_loader = DataLoader(dataset, batch_size=config.train.batch_size, num_workers=config.train.num_workers,
                                 sampler=sampler, collate_fn=_helper, drop_last=True)
    else:
        sampler = None
        data_loader = DataLoader(dataset, batch_size=config.train.batch_size, num_workers=config.train.num_workers,
                                 shuffle=True, collate_fn=_helper, drop_last=True)

    tms_model = tms_model.to(device)
    gnn_model = gnn_model.to(device)

    optimizer.zero_grad()
    for epoch in range(config.train.num_epochs):
        if is_ddp: sampler.set_epoch(epoch)

        for step, data in enumerate(data_loader):
            mri_list = data['mri']
            ef_list = data['ef']
            gt_mesh_list = data['gt_mesh']
            input_mesh_list = data['input_mesh']
            a_field_list = data['a_field']

            inputs = collate_fn(mri_list).to(device).to(torch.float32)
            laplacian_tensor_list = [get_laplacian_tensor(m, device) for m in input_mesh_list]

            _batch_size = inputs.shape[0]
            gt_vs = [torch.from_numpy(m.vertices.copy()).to(device).float() for m in gt_mesh_list]
            input_vs = [torch.from_numpy(m.vertices.copy()).to(device).float() for m in input_mesh_list]
            input_es = [torch.from_numpy(m.edges_unique.copy()).to(device).long() for m in input_mesh_list]
            a_field = [torch.from_numpy(a).to(device).float() for a in a_field_list]
            ef = [torch.from_numpy(ef_).to(device).float() for ef_ in ef_list]

            with torch.no_grad():
                for a in a_field:
                    an = torch.linalg.norm(a, dim=-1)
                    a /= an[:, None]
                    a *= torch.log(an[:, None] * 1e14)

                input_vs_aug = [random_laplacian_smoothing_torch(input_vs[j], laplacian_tensor_list[j])
                                for j in range(_batch_size)]
                input_vn = [
                    compute_vertex_normals(input_vs_aug[j],
                                           torch.from_numpy(input_mesh_list[j].faces.copy()).to(device).long())
                    for j in range(_batch_size)
                ]

            with torch.autocast(device_type='cuda', enabled=config.train.amp_enabled, dtype=torch.bfloat16):
                encoder_skips, decoder_skips = tms_model(inputs)
                skips = encoder_skips + decoder_skips

                total_loss = 0
                for j in range(_batch_size):
                    _skips = [s[j, :, :, :, :] for s in skips]
                    pred_ef = gnn_model(input_vs_aug[j], input_es[j], input_vn[j], _skips, a_field[j])

                    # chamfer loss
                    loss = loss_chamfer(input_vs[j].unsqueeze(0), gt_vs[j].unsqueeze(0), pred_ef.unsqueeze(0),
                                        ef[j].unsqueeze(0))
                    total_loss += loss

                    # laplacian loss
                    loss = loss_lap(pred_ef, laplacian_tensor_list[j])
                    total_loss += loss * config.loss.laplacian_weight

                total_loss /= _batch_size
                if local_rank == 0: print('loss: ', total_loss.item(), flush=True)

                if config.train.accumulate_grad_step != 1:
                    total_loss = total_loss / config.train.accumulate_grad_step

            total_loss.backward()
            if (step % config.train.accumulate_grad_step == config.train.accumulate_grad_step - 1 or
                    step == len(data_loader) - 1):
                # update
                torch.nn.utils.clip_grad_value_(gnn_model.parameters(), clip_value=config.train.grad_clip_value)
                optimizer.step()
                optimizer.zero_grad()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_file', '-c', type=str, default=None)
    _args = parser.parse_args()

    _config = get_default_args()
    if _args.config_file is not None:
        _config.merge_from_file(_args.config_file)
    _config.freeze()

    main(_config, vars(_args))
