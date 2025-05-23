import tinycudann as tcnn
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
import torch
class NewModel:
    def __init__(self, rvq=True):
        if rvq:
            # self.vq_scale = ResidualVQ(dim = 3, codebook_size = model.rvq_size, num_quantizers = model.rvq_num, decay = 0.8, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1).cuda()
            # self.vq_rot = ResidualVQ(dim = 4, codebook_size = model.rvq_size, num_quantizers = model.rvq_num, decay = 0.8, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1).cuda()
            self.vq_scale = ResidualVQ(dim = 3, codebook_size = 64, num_quantizers = 6, decay = 0.8, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1).cuda()
            self.vq_rot = ResidualVQ(dim = 4, codebook_size = 64, num_quantizers = 6, decay = 0.8, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1).cuda()
        
        
    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):  
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x
    

    # def training_setup(self):
        
    #     other_params = []
    #     for params in self.recolor.parameters():
    #         other_params.append(params)
    #     for params in self.mlp_head.parameters():
    #         other_params.append(params)
    #     self.optimizer_net = torch.optim.Adam(other_params, lr=0.01, eps=1e-15)
