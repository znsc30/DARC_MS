# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import mindspore
import mindspore.nn as nn
import mindspore.ops as F
from mindspore import Tensor


class IN_MLP_Rescaling4_detach_small(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-7,
        affine: bool = False,
    ) -> None:
        super(IN_MLP_Rescaling4_detach_small, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.fc_mean = nn.SequentialCell(
            nn.Dense(num_features, num_features//4,),
            nn.LayerNorm((num_features//4,)),
            nn.Dense(num_features//4, num_features,),
            nn.Sigmoid(),
        )
        self.fc_std = nn.SequentialCell(
            nn.Dense(num_features, num_features//4,),
            nn.LayerNorm((num_features//4,)),
            nn.Dense(num_features//4, num_features,),            
            nn.Sigmoid(),
        )
        self.affine = affine
        if self.affine:
            self.alpha = mindspore.Parameter(F.ones([1,num_features,1,1]))
            self.beta = mindspore.Parameter(F.zeros([1,num_features,1,1]))
        self.hist_info = mindspore.Parameter(F.randn([1, num_features,]), requires_grad=False)
        self.hist_info_update_rate = 0.01
    def forward(self, input, extra_info=None):
        bsz = input.shape[0]
        meanv = input.mean(dim=(-2,-1), keepdim=True)
        stdv = input.std(dim=(-2,-1), keepdim=True)
        # ori_m = meanv.data.view(-1)
        # ori_s = stdv.data.view(-1)
        m = meanv
        m.require_grad = False
        s = stdv
        s.require_grad = False
        if extra_info is not None:
            meanv = meanv * (self.fc_mean(m.view((bsz, -1)) + extra_info)+0.5).view(meanv.shape)
            stdv = stdv * (self.fc_std(s.view((bsz, -1)) + extra_info)+0.5).view(stdv.shape)
            if self.training:
                self.hist_info.data = self.hist_info.data * (1.0-self.hist_info_update_rate) + extra_info.detach().mean(dim=0, keepdim=True)*self.hist_info_update_rate
        else:
            meanv = meanv * (self.fc_mean(m.view((bsz, -1)) + self.hist_info.data.repeat(bsz, 1))+0.5).view(meanv.shape)
            stdv = stdv * (self.fc_std(s.view((bsz, -1)) + self.hist_info.data.repeat(bsz, 1))+0.5).view(stdv.shape)
        output = (input - meanv + self.eps) / (stdv + self.eps)
        if self.affine:
            output = output * self.alpha + self.beta
        return output

