import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import sys

from mcan.utils.make_mask import make_mask
from mcan.ops.fc import FC, MLP
from mcan.ops.layer_norm import LayerNorm
from mcan.attention.adapter import Adapter
from mcan.attention.mca import MCA_ED


class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


class Encoder(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter(__C)

        self.backbone = MCA_ED(__C)

        # Flatten to vector
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        # Classification layers
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

        self.linear = nn.Linear(1024, 300)

    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):

        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        # Classification layers
        proj_feat = lang_feat + img_feat
        # proj_feat = self.proj_norm(proj_feat)
        # proj_feat = self.proj(proj_feat)

        final_feat = self.linear(proj_feat)

        return final_feat



class Decoder(nn.Module):
    def __init__(self, __C, ans_emb, embed_size, hidden_size, vocab_size, num_layers=1):
        super(Decoder, self).__init__()

        self.__C = __C

        self.embed_size = 300
        self.hidden_size = 64
        self.vocab_size = vocab_size

        # print(len(ans_emb))

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_size
        )

        # Loading the GloVe embedding weights
        self.embedding.weight.data.copy_(torch.from_numpy(ans_emb))

        self.lstm = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=600,
            num_layers=1,
            batch_first=True
        )

        self.lstm_gen = nn.LSTMCell(
            input_size=1024,
            hidden_size=1024
        )

        self.fc_out = nn.Linear(600, self.vocab_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, ans_ix):

        lang_feat = self.embedding(ans_ix)
        test = features.unsqueeze(1)
        captions = torch.cat((features.unsqueeze(1), lang_feat), 1)

        # Get the actual lengths
        lengths = []
        for cap in ans_ix:
            cnt = 0
            for emb in cap:
                if emb != 0:
                    cnt = cnt+1
                else:
                    break
            lengths.append(cnt)

        packed = pack_padded_sequence(captions, lengths, batch_first=True)

        hiddens, _ = self.lstm(packed)
        # TODO: Check if this should be hiddens or hiddens[0]
        outputs = self.fc_out(hiddens[0])

        return outputs


class Net(nn.Module):
    
    def __init__(self, __C, pretrained_emb, token_size, answer_size, 
                    ans_emb, embed_size, hidden_size, vocab_size, num_layers):
        super(Net, self).__init__()
        self.__C = __C
        self.encoder = Encoder(self.__C, pretrained_emb, token_size, answer_size)
        self.decoder = Decoder(self.__C, ans_emb, embed_size, hidden_size, vocab_size, num_layers)
    
    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix, ans_ix):
        features = self.encoder(frcn_feat, grid_feat, bbox_feat, ques_ix)
        outputs = self.decoder(features, ans_ix)
        return outputs

    def answer_step(self):
        # TODO : complete for eval
        pass 