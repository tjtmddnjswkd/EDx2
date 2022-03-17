import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import glob
from tqdm import tqdm

class Pretrain_and_Distill_Dataset(Dataset):
    def __init__(self, data_path, tokenizer, mode, is_train):
        super(Pretrain_and_Distill_Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[MASK]'))[0]
        self.sep_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[SEP]'))[0]
        self.Train_Data, self.Train_Label, self.Val_Data, self.Val_Label = self.load_data(data_path)
        self.is_train = is_train
        self.mode = mode

    def load_data(self, data_path, val_p = 1000):
        Train_Data = []
        Val_Data = []
        Train_Label = np.array([])
        Val_Label = np.array([])
        fns = glob.glob(data_path)
        for ii in range(len(fns)):
            print('loading data: '+fns[ii])
            with open(fns[ii],'rb') as f:
                temp = pickle.load(f)
                temp_tl = np.zeros(len(temp[0])*2-val_p*2)
                temp_tl[int(len(temp_tl)/2):] = 1
                temp_vl = np.zeros(val_p*2)
                temp_vl[int(len(temp_vl)/2):] = 1
            Train_Data += temp[0][:-val_p]
            Train_Data += temp[1][:-val_p]
            Val_Data += temp[0][-val_p:]
            Val_Data += temp[1][-val_p:]
            Train_Label = np.concatenate([Train_Label,temp_tl])
            Val_Label = np.concatenate([Val_Label,temp_vl])
        
        return Train_Data, Train_Label, Val_Data, Val_Label
        
    def prepare_data(self, Train_Data, Train_Label, longest_sentence=128):
        Input_ids = np.zeros((longest_sentence))
        Token_type_ids = np.zeros((longest_sentence))
        Attention_mask = np.zeros((longest_sentence))
        Masked_lm_labels = (np.ones((longest_sentence))*-1)
        Next_sentence_label = np.zeros((1))
        temp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(Train_Data))
        if len(temp) > longest_sentence:
            sentence_length = longest_sentence
        else:
            sentence_length = len(temp)
        Input_ids[0:sentence_length] = temp[0:sentence_length]
        if self.sep_id in Input_ids:
            Token_type_ids[np.where(Input_ids==self.sep_id)[0][0]+1:sentence_length] = 1
        else:
            Token_type_ids[:] = 0
        Attention_mask[0:sentence_length] = 1
        Next_sentence_label = Train_Label
        if self.mode == 'PT' or self.mode == 'ALL':
            Input_ids, Masked_lm_labels = self.tokenizer.Masking(Input_ids, Masked_lm_labels)
        return (Input_ids, Token_type_ids, Attention_mask, Masked_lm_labels, Next_sentence_label)
    
    def __len__(self):
        if self.is_train:
            return len(self.Train_Data)
        else:
            return len(self.Val_Data)
    
    def __getitem__(self, index):
        if self.is_train:
            self.INPUT = self.prepare_data(self.Train_Data[index], self.Train_Label[index])
        else: self.INPUT = self.prepare_data(self.Val_Data[index], self.Val_Label[index])
        
        return torch.tensor(self.INPUT[0]).long(), \
                torch.tensor(self.INPUT[1]).long(), \
                torch.tensor(self.INPUT[2]).long(), \
                torch.tensor(self.INPUT[3]).long(), \
                torch.tensor(self.INPUT[4]).long()
                
