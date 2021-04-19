import os, sys
import numpy as np
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from pathlib import Path

from vqa_loader import DataSet
from config.model_cnfgs import Cfgs
from net import Net
from mcan.utils.optim import get_optim, adjust_lr
from loss.bert_score_loss import BertLossFunc

now = datetime.now()

__C = Cfgs()

__C.VERSION = str(int(datetime.timestamp(now)))

dataset = DataSet()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
Path('models/bs').mkdir(parents=True, exist_ok=True)

data_size = dataset.data_size
ques_emb = dataset.ques_emb
ques_token_size = dataset.ques_token_size
answer_size = dataset.ans_token_size
ans_emb = dataset.ans_emb
embed_size = 128
hidden_size = 128
vocab_size = answer_size
num_layers = 1

net = Net(__C, ques_emb, ques_token_size, answer_size, 
                    ans_emb, embed_size, hidden_size, vocab_size, num_layers)
net.train()
net.cuda()

'''Optimizers'''
optim = get_optim(__C, net, data_size)
start_epoch = 0

loss_sum = 0
named_params = list(net.named_parameters())
grad_norm = np.zeros(len(named_params))

for epoch in range(start_epoch, __C.MAX_EPOCH):

    # Learning Rate Decay
    if epoch in __C.LR_DECAY_LIST:
        adjust_lr(optim, __C.LR_DECAY_R)
    
    for step, (
        frcn_feat_iter,
        grid_feat_iter,
        bbox_feat_iter,
        ques_ix_iter,
        ans_ix_iter
    ) in enumerate(dataloader):

        optim.zero_grad()
        
        '''Sort according to lengths of ans_ix'''
        lengths = []
        for cap in ans_ix_iter:
            cnt = 0
            for emb in cap:
                if emb != 0:
                    cnt = cnt+1
                else:
                    break
            lengths.append(cnt)
        lengths = torch.tensor(lengths)
        lengths, sort_ix = lengths.sort(dim=0, descending=True)

        frcn_feat_iter = frcn_feat_iter[sort_ix]
        grid_feat_iter = grid_feat_iter[sort_ix]
        bbox_feat_iter = bbox_feat_iter[sort_ix]
        ques_ix_iter = ques_ix_iter[sort_ix]
        ans_ix_iter = ans_ix_iter[sort_ix]
        
        y = pack_padded_sequence(ans_ix_iter, lengths, batch_first=True)[0]
        
        '''Send data to network'''
        y1 = net(frcn_feat_iter, grid_feat_iter, bbox_feat_iter, ques_ix_iter, ans_ix_iter)
        
        '''Bert Score loss function'''
        criterion = BertLossFunc()
        loss = criterion(y1, ans_ix_iter)
        loss.backward()
        loss_sum += loss.cpu().data.numpy()

        print('')
        # print('Loss : '+str(loss.item()))
        # print('---------------------------------------')
        print('\r [EPOCH : %d] [STEP : %d/%d] [LOSS FUNCTION : %s] [LOSS : %s]' %(
            epoch+1,
            step+1,
            int(data_size/__C.BATCH_SIZE),
            'BS',
            str(loss.item()))
        )

        optim.step()
        
    '''Save Model'''
    epoch_final = epoch+1
    state = {
        'state_dict': net.state_dict(),
        'optimizer': optim.optimizer.state_dict(),
        'lr_base': optim.lr_base,
        'epoch': epoch_final
    }
    torch.save(
        state,
        'models/bs' +
        '/epoch' + str(epoch_final) +
        '.pkl'
    )