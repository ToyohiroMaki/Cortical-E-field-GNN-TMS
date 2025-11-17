import os
from typing import Dict

import numpy as np
import nibabel as nib
import trimesh
import meshio
import pymeshfix as mf
import pymeshlab


__all__ = [
    'meshfix', 'simplify_mesh', 'save_gmsh', 'load_gii'
]


def meshfix(mesh: trimesh.Trimesh, verbose: bool = False) -> trimesh.Trimesh:
    m = mf.MeshFix(mesh.vertices, mesh.faces)
    m.repair(verbose=verbose)
    m = trimesh.Trimesh(m.v, m.f, process=True)
    m.fix_normals()
    return m


def simplify_mesh(mesh: trimesh.Trimesh, target_perc: float):
    assert 0 <= target_perc <= 1

    new_mesh = pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(new_mesh)
    ms.meshing_decimation_quadric_edge_collapse(targetperc=target_perc,
                                                preserveboundary=True,
                                                preservetopology=True)
    m = ms.current_mesh()
    m = trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=True)
    m.fix_normals()
    return m


def load_gii(path: str, process: bool = True) -> trimesh.Trimesh:
    assert os.path.splitext(path)[-1] == '.gii'
    hemi_data = nib.load(path)
    v, f = hemi_data.agg_data()
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=process)
    return mesh


def save_gmsh(path: str, mesh: trimesh.Trimesh, point_data: Dict[str, np.ndarray] = None):
    cells = [('triangle', mesh.faces)]
    mesh = meshio.Mesh(points=mesh.vertices, cells=cells, point_data=point_data)
    meshio.write(path, mesh, file_format='gmsh')
