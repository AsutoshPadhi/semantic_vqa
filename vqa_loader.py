import numpy as np
import glob, json, re, os, sys
import en_vectors_web_lg
from dataset_loader import DatasetLoader

class DataSet(DatasetLoader):
    
    def __init__(self):
        super(DataSet, self).__init__()
        
        '''Load paths'''
        self.feat_path_list = glob.glob('data/feat/train2014/'+'/*.npz')
        self.ques_list = json.load(open('data/raw/questions.json'))
        self.ans_list = json.load(open('data/raw/answers.json'))

        # print(self.ques_list[0])
        # print(self.ans_list[0])
        
        self.data_size = self.ans_list.__len__()

        '''{image id} -> {image feature absolutely path}'''
        self.iid_to_feat_path = self.img_feat_path_load(self.feat_path_list)

        '''{question id} -> {question}'''
        self.qid_to_ques = self.ques_load(self.ques_list)

        '''Tokenize questions'''
        self.ques_token_to_ix, self.ques_emb = self.tokenize_ques(self.ques_list)
        self.ques_token_size = self.ques_token_to_ix.__len__()
        print(' ========== Question token vocab size:', self.ques_token_size)

        '''Tokenize answers'''
        self.ans_token_to_ix, self.ans_emb = self.tokenize_ans(self.ans_list)
        self.ans_ix_to_token = {v: k for k, v in self.ans_token_to_ix.items()}
        self.ans_token_size = self.ans_token_to_ix.__len__()
        print(' ========== Answer Gen token vocab size:', self.ans_token_size)

        # print(self.ques_token_to_ix)
        # print(self.ans_token_to_ix)


    def img_feat_path_load(self, path_list):
        iid_to_path = {}

        for ix, path in enumerate(path_list):
            iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
            # print(iid)
            iid_to_path[iid] = path

        return iid_to_path


    def ques_load(self, ques_list):
        qid_to_ques = {}

        for ques in ques_list:
            qid = str(ques['question_id'])
            qid_to_ques[qid] = ques

        return qid_to_ques


    def tokenize_ques(self, stat_ques_list):
        ques_token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        spacy_tool = en_vectors_web_lg.load()
        
        pretrained_emb = []
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)
        pretrained_emb.append(spacy_tool('CLS').vector)

        # print('\nQuestion: '+stat_ques_list[0])
        for ques in stat_ques_list:
            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ques['question'].lower()
            ).replace('-', ' ').replace('/', ' ').split()

            for word in words:
                if word not in ques_token_to_ix:
                    ques_token_to_ix[word] = len(ques_token_to_ix)
                    pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)
        
        """ token_to_ix is a word:index dictionary; pretrained embeddings contains embeddings of all the words present in all the questions """
        return ques_token_to_ix, pretrained_emb


    def tokenize_ans(self, stat_ans_list):
        ans_token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
            'SEP': 3
        }

        spacy_tool = en_vectors_web_lg.load()
        
        ans_emb = []
        ans_emb.append(spacy_tool('PAD').vector)
        ans_emb.append(spacy_tool('UNK').vector)
        ans_emb.append(spacy_tool('CLS').vector)
        ans_emb.append(spacy_tool('SEP').vector)

        for ans in stat_ans_list:
            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ans['multiple_choice_answer'].lower()
            ).replace('-', ' ').replace('/', ' ').split()

            for word in words:
                if word not in ans_token_to_ix:
                    ans_token_to_ix[word] = len(ans_token_to_ix)
                    ans_emb.append(spacy_tool(word).vector)

        ans_emb = np.array(ans_emb)
        
        """ token_to_ix is a word:index dictionary; pretrained embeddings contains embeddings of all the words present in all the questions """
        return ans_token_to_ix, ans_emb


    def load_ques_ans(self, idx):
        ans = self.ans_list[idx]
        ques = self.qid_to_ques[str(ans['question_id'])]
        iid = str(ans['image_id'])
        
        # Process question
        ques_ix_iter = self.proc_ques(ques, self.ques_token_to_ix, max_token=14)
        # Process answer gen
        ans_ix_iter = self.proc_ans_gen(ans, self.ans_token_to_ix, max_token=14)

        return ques_ix_iter, ans_ix_iter, iid, ques['question'], ans['multiple_choice_answer']


    def proc_ques(self, ques, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix


    def proc_ans_gen(self, ans, ans_token_to_ix, max_token):

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ans['multiple_choice_answer'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        ans_ix = np.zeros(max_token, np.int64)
        
        ans_ix[0] = ans_token_to_ix['CLS']
        
        for ix, word in enumerate(words):
            if ix + 1 == max_token:
                break
            if word in ans_token_to_ix:
                ans_ix[ix+1] = ans_token_to_ix[word]
            else:
                ans_ix[ix+1] = ans_token_to_ix['UNK']

        for i, num in enumerate(ans_ix):
            if num == 0:
                ans_ix[i] = ans_token_to_ix['SEP']
                break
        # print(ans_ix)
        # print('ans ix size = '+str(len(ans_ix)))
        # sys.exit('vqa_loader 189')
        return ans_ix


    def load_img_feats(self, idx, iid):
        frcn_feat = np.load(self.iid_to_feat_path[iid])
        frcn_feat_x = frcn_feat['x'].transpose((1, 0))
        frcn_feat_iter = self.proc_img_feat(frcn_feat_x, img_feat_pad_size=100)

        bbox_feat_iter = self.proc_img_feat(
            self.proc_bbox_feat(
                frcn_feat['bbox'],
                (frcn_feat['image_h'], frcn_feat['image_w'])
            ),
            img_feat_pad_size=100
        )
        grid_feat_iter = np.zeros(1)

        return frcn_feat_iter, grid_feat_iter, bbox_feat_iter


    def proc_img_feat(self, img_feat, img_feat_pad_size):
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]

        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return img_feat


    def proc_bbox_feat(self, bbox, img_shape):
        # if self.__C.BBOX_NORMALIZE:
        if True:
            bbox_nm = np.zeros((bbox.shape[0], 4), dtype=np.float32)

            bbox_nm[:, 0] = bbox[:, 0] / float(img_shape[1])
            bbox_nm[:, 1] = bbox[:, 1] / float(img_shape[0])
            bbox_nm[:, 2] = bbox[:, 2] / float(img_shape[1])
            bbox_nm[:, 3] = bbox[:, 3] / float(img_shape[0])
            return bbox_nm
        # bbox_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / float(img_shape[0] * img_shape[1])

        return bbox


if __name__ == '__main__':
    c = DataSet()