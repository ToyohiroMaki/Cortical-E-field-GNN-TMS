import torch
from torch import nn, Tensor
from torch.nn import functional as F
from pytorch3d.ops import knn_points


__all__ = [
    'ChamferLossEF', 'LaplacianLossEF', 'chamfer_loss_ef'
]


def chamfer_loss_ef(pred_vertices: Tensor, trg_vertices: Tensor, pred_e: Tensor, trg_e: Tensor):
    """
    :param pred_vertices: N x V x 3
    :param trg_vertices: N x V x 3
    """
    with torch.no_grad():
        trg2pred_idx = knn_points(trg_vertices, pred_vertices, K=1).idx
        pred2trg_idx = knn_points(pred_vertices, trg_vertices, K=1).idx

    loss_trg2pred = F.mse_loss(trg_e, pred_e[:, trg2pred_idx.squeeze(), :])
    loss_pred2trg = F.mse_loss(pred_e, trg_e[:, pred2trg_idx.squeeze(), :])

    loss = loss_trg2pred.mean() + loss_pred2trg.mean()
    return loss


class ChamferLossEF(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_vertices: Tensor, trg_vertices: Tensor, pred_e: Tensor, trg_e: Tensor):
        """
        :param pred_vertices: N x V x 3
        :param trg_vertices: N x V x 3
        """
        return chamfer_loss_ef(pred_vertices, trg_vertices, pred_e, trg_e)


class LaplacianLossEF(nn.Module):
    def __init__(self, lap_kind='nrw'):
        """
        lap_kind:
            nrw -> I - D^{-1}A
            trimesh -> D^{-1}A
        """
        super().__init__()
        assert lap_kind in ['nrw', 'trimesh']
        self.lap_kind = lap_kind

    def forward(self, pred_ef: Tensor, lap: Tensor):
        # "addmm_sparse_cuda" not implemented for 'BFloat16'
        with torch.amp.autocast(device_type='cuda', enabled=False):
            pred_ef = pred_ef.float()
            lap = lap.float()

            if self.lap_kind == 'nrw':
                loss = torch.norm(lap.mm(pred_ef), dim=1)
            elif self.lap_kind == 'trimesh':
                loss = torch.norm(pred_ef - lap.mm(pred_ef), dim=1)
            else:
                raise NotImplementedError
        return loss.mean()
