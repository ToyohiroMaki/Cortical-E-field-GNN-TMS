import os
from argparse import ArgumentParser

import numpy as np
import nibabel as nib
import torch
import trimesh

from models.gnn import GNN
from models.unet import ResidualUNet
from utils import mapping_numpy as mnp
from utils.torch_utils import restore_model
from utils.mesh_utils import load_gii, simplify_mesh, meshfix, save_gmsh
from utils.mri_utils import adjust_orientation
from utils.field import MFieldFMM3D
from utils.mesh_utils_torch import compute_vertex_normals
from config.config import get_default_args


def load_sample_data():
    mesh = load_gii('sample/lh.pial.gii')
    mesh = meshfix(mesh)
    mesh = simplify_mesh(mesh, target_perc=60000 / mesh.vertices.shape[0])
    mesh = trimesh.smoothing.filter_laplacian(mesh)

    t1 = nib.load('sample/T1.nii.gz')
    t1 = adjust_orientation(t1)
    affine = t1.affine
    t1 = t1.get_fdata().squeeze()
    t1 = np.clip(t1 / t1.max(), 0, 1)

    v_voxel = mnp.subject2voxel(np.copy(mesh.vertices), affine=affine)
    mesh.vertices = v_voxel
    return t1, affine, mesh


def compute_c(a_field):
    an = torch.linalg.norm(a_field, dim=-1)
    c = a_field / an[:, None]
    c *= torch.log(an[:, None] * 1e14)
    return c


def main(args: dict):
    config = get_default_args()

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

    state_dict = torch.load(args['unet'], map_location='cpu', weights_only=True)
    restore_model(state_dict, tms_model)
    state_dict = torch.load(args['gnn'], map_location='cpu', weights_only=True)
    restore_model(state_dict, gnn_model)

    tms_model.eval()
    gnn_model.eval()

    device = torch.device('cuda:0') if args['cuda'] else torch.device('cpu')
    tms_model = tms_model.to(device)
    gnn_model = gnn_model.to(device)

    mri, affine, input_mesh = load_sample_data()
    matsimnibs = np.load('sample/tmp_matsimnibs.npy')
    matsimnibs = matsimnibs[0]

    module = MFieldFMM3D(kind='A')
    v_coil = mnp.voxel2coil(np.copy(input_mesh.vertices), matsimnibs=matsimnibs, affine=affine)
    a_field = module.compute_field(points=v_coil)
    a_field = mnp.coil2voxel(a_field, matsimnibs=matsimnibs, affine=affine, no_translation=True)

    a_field_t = torch.from_numpy(a_field).to(device).float()
    v_t = torch.from_numpy(input_mesh.vertices.copy()).to(device).float()
    e_t = torch.from_numpy(input_mesh.edges_unique.copy()).to(device).long()
    f_t = torch.from_numpy(input_mesh.faces.copy()).to(device).long()
    mri_t = torch.from_numpy(mri).to(device).float()

    with torch.no_grad():
        c_t = compute_c(a_field_t)
        vn = compute_vertex_normals(v_t, f_t)

        encoder_skips, decoder_skips = tms_model(mri_t.unsqueeze(0).unsqueeze(0))
        skips = encoder_skips + decoder_skips

        _skips = [s[0, :, :, :, :] for s in skips]
        pred_ef = gnn_model(v_t, e_t, vn, _skips, c_t)

    pred_ef = pred_ef.detach().cpu().numpy().squeeze()
    os.makedirs(args['out_dir'], exist_ok=True)
    save_gmsh(os.path.join(args['out_dir'], 'result.msh'), input_mesh,
              point_data={'E_mag': np.linalg.norm(pred_ef, axis=-1)})


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gnn', '-g', type=str, default='sample/model-epoch80-gnn.pt')
    parser.add_argument('--unet', '-u', type=str, default='sample/model-epoch80-unet.pt')
    parser.add_argument('--out_dir', type=str, default='sample_out')
    parser.add_argument('--cuda', action='store_true')
    _args = parser.parse_args()

    main(vars(_args))
