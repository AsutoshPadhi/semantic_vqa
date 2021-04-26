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
from queue import PriorityQueue


SOS_token = 2
EOS_token = 3


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

        hiddens, _ = self.lstm(captions)
        # TODO: Check if this should be hiddens or hiddens[0]
        outputs = self.fc_out(hiddens)

        outputs = self.softmax(outputs)

        return outputs


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


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

    def beam_search(self, ans_ix, features):
        
        beam_width = 3
        topk = 1
        decoded_batch = []

        for idx in range(self.__C.BATCH_SIZE):
            decoder_hidden = features[idx].unsqueeze(0)
            decoder_input = torch.LongTensor([[SOS_token]])

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000:
                    break
                # fetch the best node
                score, n = nodes.get()
                decoder_input = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == EOS_token and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch