from torch.utils.data import Dataset
import torch
import sys

class DatasetLoader(Dataset):
    
    def __init__(self):
        
        '''Load paths'''
        self.feat_path_list = None
        self.ques_list = None
        self.ans_list = None
        self.data_size = None

        self.ques_token_to_ix = None
        self.ques_emb = None
        self.ques_token_size = None
        self.ans_token_to_ix = None
        self.ans_emb = None
        self.ans_token_size = None


    def __len__(self):
        return self.ans_token_size


    def __getitem__(self, idx):
        ques_ix_iter, ans_ix_iter, iid, ques, ans = self.load_ques_ans(idx)
        frcn_feat_iter, grid_feat_iter, bbox_feat_iter = self.load_img_feats(idx, iid)
        return frcn_feat_iter, grid_feat_iter, bbox_feat_iter, ques_ix_iter, ans_ix_iter