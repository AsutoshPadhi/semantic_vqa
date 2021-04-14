import glob, os, sys, json
from pathlib import Path
from shutil import copyfile
import numpy as np
from tqdm import tqdm

class CreateDataset():
    def __init__(self):
        
        '''Init source folder names'''
        self.root_folder = '../openvqa-master/data/vqa/'
        self.feat_folder = self.root_folder+'feats/'
        self.raw_folder = self.root_folder+'raw/'

        '''Load the answers'''
        self.annotations_train_file = self.raw_folder+'/v2_mscoco_train2014_annotations.json'
        self.annotations_list = json.load(open(self.annotations_train_file))['annotations']
        self.multi_word_annotatons_list = []

        '''Load the questions'''
        self.ques_train_file = self.raw_folder+'/v2_OpenEnded_mscoco_train2014_questions.json'
        self.ques_list_full = json.load(open(self.ques_train_file))['questions']
        self.ques_list = []
        self.ques_dict = {}
        # print(self.ques_list_full)
        for row in self.ques_list_full:
            self.ques_dict[row['question_id']] = row
        # for k,v in self.ques_dict.items():
        #     print(k, v)

        '''Create empty folders'''
        Path('data/feat/train2014').mkdir(parents=True, exist_ok=True)
        self.new_feat_folder = 'data/feat/train2014/'

    def create_multi_word_ans_dataset(self):
        
        src_folder = self.feat_folder+'train2014/'
        dst_folder = self.new_feat_folder

        '''Create a list of multiword answers'''
        for row in self.annotations_list:
            if(len(row['multiple_choice_answer'].split()) > 1):
                self.multi_word_annotatons_list.append(row)
        with open('./data/raw/answers.json', 'w') as fout:
            json.dump(self.multi_word_annotatons_list , fout)
        
        '''Create DataSet'''
        for row in tqdm(self.multi_word_annotatons_list):
            '''Copy Images'''
            image_id = str(row['image_id']).zfill(12)
            src = src_folder+'/COCO_train2014_'+image_id+'.jpg.npz'
            dst = dst_folder+'/COCO_train2014_'+image_id+'.jpg.npz'
            copyfile(src, dst)
            '''Copy Questions'''
            question_id = row['question_id']
            self.ques_list.append(self.ques_dict[question_id])
        
        with open('./data/raw/questions.json', 'w') as fout:
            json.dump(self.ques_list , fout)


if __name__ == '__main__':
    c = CreateDataset().create_multi_word_ans_dataset()