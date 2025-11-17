from yacs.config import CfgNode


__all__ = [
    'get_default_args'
]


_C = CfgNode()

_C.model = CfgNode()
_C.model.input_img_size = [197, 233, 189]
# parameter of u-net
_C.model.encoder_channels = [16, 32, 64, 128, 256]
_C.model.decoder_channels = [64, 32, 16, 8]
# parameter of gnn
_C.model.graph_channels = [256, 64, 64, 64, 64]
_C.model.agg_indices = [[3, 4, 5, 6], [2, 3, 6, 7], [1, 2, 7, 8], [0, 1, 7, 8]]
_C.model.residual_block_num = 4
_C.model.graph_conv_block_num = 4


_C.loss = CfgNode()
_C.loss.laplacian_weight = 0.001

_C.optimizer = CfgNode()
_C.optimizer.lr = 1e-4
_C.optimizer.lr_unet = 1e-3

_C.data = CfgNode()

_C.train = CfgNode()
_C.train.batch_size = 3
_C.train.num_workers = 1
_C.train.amp_enabled = True
_C.train.accumulate_grad_step = 1
_C.train.grad_clip_enabled = True
_C.train.grad_clip_value = 0.5
_C.train.num_epochs = 100

# _C.log = CfgNode()
# _C.log.logdir = './debug'


def get_default_args():
    return _C.clone()
