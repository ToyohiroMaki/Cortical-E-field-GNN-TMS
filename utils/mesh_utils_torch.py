import numpy as np

import trimesh
import torch
from torch import Tensor


__all__ = [
    'compute_vertex_normals', 'compute_face_normals', 'random_laplacian_smoothing_torch', 'get_laplacian_tensor'
]


def compute_face_normals(v: Tensor, f: Tensor) -> Tensor:
    v0 = v[f[:, 0], :]
    v1 = v[f[:, 1], :]
    v2 = v[f[:, 2], :]

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_normals = face_normals / torch.norm(face_normals, dim=-1, keepdim=True)
    return face_normals


def compute_vertex_normals(v: Tensor, f: Tensor) -> Tensor:
    face_normals = compute_face_normals(v, f)

    vertex_normal = torch.zeros_like(v)
    vertex_normal = vertex_normal.index_add(0, f[:, 0], face_normals)
    vertex_normal = vertex_normal.index_add(0, f[:, 1], face_normals)
    vertex_normal = vertex_normal.index_add(0, f[:, 2], face_normals)
    vertex_normal = torch.nn.functional.normalize(vertex_normal, dim=1)
    return vertex_normal


def random_laplacian_smoothing_torch(v: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
    # laplacian: D^{-1} A
    w = torch.randn(1).to(v.device)
    v_new = v + w * (laplacian.mm(v) - v)
    return v_new


def get_laplacian_tensor(mesh: trimesh.Trimesh, device: torch.device) -> Tensor:
    lap = trimesh.smoothing.laplacian_calculation(mesh)  # D_inv A
    indices = torch.from_numpy(np.stack([lap.row, lap.col])).to(torch.long)
    values = torch.from_numpy(lap.data).to(torch.float32)
    lap_torch = torch.sparse_coo_tensor(indices, values, device=device)
    return lap_torch  # D_inv A
