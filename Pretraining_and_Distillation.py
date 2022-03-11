#!/usr/bin/env python
# coding: utf-8 %%

# %%

import numpy as np
import pickle
import copy
import torch as t
import re
import torch
import numpy as np
import time
import argparse
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import random
from exBERT import BertTokenizer, BertAdam
from dataset import Pretrain_and_Distill_Dataset
# %%

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", required = True, type = str, help='PT or DT or ALL')
ap.add_argument("-e", "--epochs", required = True, type = int, help='number of training epochs')
ap.add_argument("-b", "--batchsize", required = True, type = int, help='training batchsize')
ap.add_argument("-sp","--save_path",required = True, type = str, help='path to storaget the loss table, stat_dict')
ap.add_argument('-dv','--device',required = True, type = int, nargs='+',help='gpu id for the training, ex [-dv 0 12 3]')
ap.add_argument('-lr','--learning_rate',required = True, type = float, help='learning rate , google use 1e-04')
ap.add_argument('-str','--strategy',required = True, type = str, help='choose a strategy from [exBERT], [sciBERT], [bioBERT]')
ap.add_argument('-config','--config',required = True, type = str, nargs = '+', help='dir to the config file')
ap.add_argument('-vocab','--vocab',required = True, type = str, help='path to the vocab file for tokenization')
ap.add_argument('-pm_p','--pretrained_model_path',default = None, type = str, help='path to the pretrained_model stat_dict (torch state_dict)')
ap.add_argument('-dp','--data_path',required = True, type = str, help='path to data ')
ap.add_argument('-ls','--longest_sentence', required = True, type = int, help='set the limit of the sentence lenght, recommand the same to the -dt')
ap.add_argument('-wp', '--warmup', default=-1, type=float, help='portion of all training itters to warmup, -1 means not using warmup')
ap.add_argument('-t_ex_only','--train_extension_only', default=True, type=bool, help='train only the extension module')
args = vars(ap.parse_args())
for ii, item in enumerate(args):
    print(item+': '+str(args[item]))
## set device
if args['device'] == [-1]:
    device = 'cpu'
    device_ids = 'cpu'
else:
    device_ids = args['device']
    device = 'cuda:'+str(device_ids[0])
    print('training with GPU: '+str(device_ids))


class pre_train_BertTokenizer(BertTokenizer):
    def __init__(self, vocab_file, **kwargs):
        '''
        
        '''
        super(pre_train_BertTokenizer,self).__init__(vocab_file)
        self.mask_id = self.convert_tokens_to_ids(self.tokenize('[MASK]'))[0]
        self.sep_id = self.convert_tokens_to_ids(self.tokenize('[SEP]'))[0]

    def Masking(self, Input_ids, Masked_lm_labels):
        copyInput_ids = copy.deepcopy(Input_ids)
        rd_1 = np.random.random(Input_ids.shape)
        rd_1[0] = 0
        Masked_lm_labels[rd_1>0.85] = Input_ids[rd_1>0.85]
        Input_ids[rd_1>=0.88] = self.mask_id
        Input_ids[(rd_1>=0.865)*(rd_1<0.88)] = (np.random.rand(((rd_1>=0.865)*(rd_1<0.88)*1).sum())*len(self.vocab)).astype(int)
        Input_ids[copyInput_ids==0] = 0
        Masked_lm_labels[copyInput_ids==0] = -1
        return Input_ids, Masked_lm_labels

tok = pre_train_BertTokenizer(args['vocab'])
train_dataset = Pretrain_and_Distill_Dataset(args['data_path'], tok, is_train=True)
val_dataset = Pretrain_and_Distill_Dataset(args['data_path'], tok, is_train=False)
print('done data preparation')
print('data number: '+str(len(train_dataset)))

if args['strategy'] == 'exBERT':
    from exBERT import BertForPreTraining, BertConfig
    bert_config_1 = BertConfig.from_json_file(args['config'][0])
    bert_config_2 = BertConfig.from_json_file(args['config'][1])
    print("Building PyTorch model from configuration: {}".format(str(bert_config_1)))
    print("Building PyTorch model from configuration: {}".format(str(bert_config_2)))
    model = BertForPreTraining(bert_config_1, bert_config_2)
else:
    from exBERT import BertForPreTraining, BertConfig
    bert_config_1 = BertConfig.from_json_file(args['config'][0])
    print("Building PyTorch model from configuration: {}".format(str(bert_config_1)))
    model = BertForPreTraining(bert_config_1)

## load pre-trained model
# if args['pretrained_model_path'] is not None:
#     stat_dict = t.load(args['pretrained_model_path'], map_location='cpu')
#     for key in list(stat_dict.keys()):
#         stat_dict[key.replace('gamma', 'weight').replace('beta', 'bias')] = stat_dict.pop(key)
#     # import pdb;pdb.set_trace()
#     model.load_state_dict(stat_dict, strict=False) 
    
sta_name_pos = 0
if device is not 'cpu':
    if len(device_ids)>1:
        model = nn.DataParallel(model,device_ids=device_ids)
        sta_name_pos = 1
    model.to(device)

if args['strategy'] == 'exBERT':
    if args['train_extension_only']:
        for ii,item in enumerate(model.named_parameters()):
            item[1].requires_grad=False
            if 'ADD' in item[0]:
                item[1].requires_grad = True
            if 'pool' in item[0]:
                item[1].requires_grad=True
            if item[0].split('.')[sta_name_pos]!='bert':
                item[1].requires_grad=True

print('The following part of model is goinig to be trained:')
for ii, item in enumerate(model.named_parameters()):
    if item[1].requires_grad:
        print(item[0])

lr = args['learning_rate']
param_optimizer = list(model.named_parameters())
param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

num_epoc = args['epochs']
batch_size = args['batchsize']
longest_sentence = args['longest_sentence']
total_batch_num = int(np.ceil(len(train_dataset)/batch_size))
optimizer = BertAdam(optimizer_grouped_parameters, lr=lr) # , warmup=args['warmup'], t_total=total_batch_num
all_data_num = len(train_dataset)

train_los_table = np.zeros((num_epoc,total_batch_num))
val_los_table = np.zeros((num_epoc,total_batch_num))
best_loss = float('inf')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2*len(args['device']))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2*len(args['device']))

def process_batch(INPUT, is_train = True):
    if is_train:
        model.train()
        optimizer.zero_grad()
    Input_ids = t.tensor(INPUT[0]).long().to(device)
    Token_type_ids = t.tensor(INPUT[1]).long().to(device)
    Attention_mask = t.tensor(INPUT[2]).long().to(device)
    Masked_lm_labels = t.tensor(INPUT[3]).long().to(device)
    Next_sentence_label = t.tensor(INPUT[4]).long().to(device)
    loss1 = model(Input_ids,
          token_type_ids = Token_type_ids,
          attention_mask = Attention_mask,
          masked_lm_labels = Masked_lm_labels,
          next_sentence_label = Next_sentence_label
         )
    if is_train:
        loss1.sum().unsqueeze(0).backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss1.sum().data

save_id = 0
print_every_ndata = int(all_data_num/batch_size/20) ##output log every 0.5% of data of an epoch is processed
try:
    for epoc in range(num_epoc):
        t2 = time.time()
        train_loss = 0
        val_loss = 0
        model.train()
        for batch_ind, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
            Input_ids = batch[0].to(device)
            Token_type_ids = batch[1].to(device)
            Attention_mask = batch[2].to(device)
            Masked_lm_labels = batch[3].to(device)
            Next_sentence_label = batch[4].to(device)

            optimizer.zero_grad()

            loss1 = model(Input_ids,
                token_type_ids = Token_type_ids,
                attention_mask = Attention_mask,
                masked_lm_labels = Masked_lm_labels,
                next_sentence_label = Next_sentence_label)
            loss1.sum().unsqueeze(0).backward()
            optimizer.step()
            
            train_log = loss1.sum().data

            train_los_table[epoc,batch_ind] = train_log
            train_loss+=train_log
            if batch_ind>0 and batch_ind % print_every_ndata ==0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttrain_Loss: {:.5f} \tval_Loss: {:.5f} time: {:.4f} \t lr:{:.6f}'.format(
                        epoc,
                        batch_ind*batch_size,
                        all_data_num, 100 * (batch_ind*batch_size) / all_data_num,
                        train_loss/print_every_ndata/batch_size,val_loss/print_every_ndata/batch_size,time.time()-t2 ,
                        optimizer.get_lr()[0]))
                train_loss = 0
                val_loss = 0
                if not os.path.exists(args['save_path']):
                    os.mkdir(args['save_path'])
                with open(args['save_path']+'/loss.pkl','wb') as f:
                    pickle.dump([train_los_table,val_los_table,args],f)
        if len(device_ids)>1:
            t.save(model.module.state_dict(),args['save_path']+'/state_dic_'+args['strategy']+'_'+str(epoc))
        else:
            t.save(model.state_dict(),args['save_path']+'/state_dic_'+args['strategy']+'_'+str(epoc))
        with open(args['save_path']+'/loss.pkl','wb') as f:
            pickle.dump([train_los_table,val_los_table,args],f)
        model.eval()
        with t.no_grad():
            for batch_ind, batch in enumerate(val_dataloader):
                Input_ids = batch[0].to(device)
                Token_type_ids = batch[1].to(device)
                Attention_mask = batch[2].to(device)
                Masked_lm_labels = batch[3].to(device)
                Next_sentence_label = batch[4].to(device)

                loss1 = model(Input_ids,
                token_type_ids = Token_type_ids,
                attention_mask = Attention_mask,
                masked_lm_labels = Masked_lm_labels,
                next_sentence_label = Next_sentence_label)
            
                val_log = loss1.sum().data
                val_loss+=val_log
        print('Val_loss: '+str(val_loss/(batch_ind+1)))
        if val_loss.data < best_loss:
            if len(device_ids)>1:
                t.save(model.module.state_dict(),args['save_path']+'/Best_stat_dic_'+args['strategy'])
            else:
                t.save(model.state_dict(),args['save_path']+'/Best_stat_dic_'+args['strategy'])
            best_loss = val_loss.data
            print('update!!!!!!!!!!!!')
except KeyboardInterrupt:
    print('saving stat_dict and loss table')
    with open(args['save_path']+'/kbstop_loss.pkl','wb') as f:
        pickle.dump([train_los_table,val_los_table,args],f)
    t.save(model.state_dict(),args['save_path']+'/kbstop_stat_dict')




