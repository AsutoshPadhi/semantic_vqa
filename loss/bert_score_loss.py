from numpy.core.numeric import require
import torch
import torch.nn as nn
from torch.autograd import Variable
from bert_score import score
from utils.reverse_padded_sequence import ReversePaddedSequence as RPS
from vqa_loader import DataSet

dataset = DataSet()

class BertLossFunc(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.ans_ix_to_token = dataset.ans_ix_to_token

        self.cands = []
        self.refs = []

    def forward(self, y1, y):
        
        self.y1 = (torch.argmax(y1, 2)).type(torch.DoubleTensor)
        self.y = y.type(torch.DoubleTensor)
        
        self.y1 = Variable(self.y1, requires_grad=True)
        self.y = Variable(self.y, requires_grad=True)

        for row in self.y:
            sent = ''
            for ix in row.tolist():
                sent = sent + self.ans_ix_to_token[ix] + ' '
            self.cands.append(sent)

        for row in self.y1:
            sent = ''
            for ix in row.tolist():
                sent = sent + self.ans_ix_to_token[ix] + ' '
            self.refs.append(sent)

        P, R, F1 = score(self.cands, self.refs, lang="en", verbose=False)
        P = Variable(P, requires_grad=True)
        loss = torch.mean(P)
        
        return loss