import torch.nn as nn
from .feat_filter import feat_filter

class BaseAdapter(nn.Module):
    def __init__(self, __C):
        super(BaseAdapter, self).__init__()
        self.__C = __C
        
        self.vqa_init(__C)

    def vqa_init(self, __C):
        raise NotImplementedError()

    def forward(self, frcn_feat, grid_feat, bbox_feat):
        feat_dict = feat_filter(self.__C.DATASET, frcn_feat, grid_feat, bbox_feat)

        return self.vqa_forward(feat_dict)

    def vqa_forward(self, feat_dict):
        raise NotImplementedError()
