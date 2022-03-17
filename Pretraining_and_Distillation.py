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
from transformers.models.bert.tokenization_bert import BertTokenizer
from exBERT import BertAdam
from dataset import Pretrain_and_Distill_Dataset
import logging
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
ap.add_argument('-config_s','--config_s', type = str, nargs = '+', help='path to the config file of student')
ap.add_argument('-vocab','--vocab',required = True, type = str, help='path to the vocab file for tokenization')
ap.add_argument('-vocab_bert','--vocab_bert', type = str, help='path to the vocab file of bert for tokenization')
ap.add_argument('-pm_p','--pretrained_model_path',default = None, type = str, help='path to the pretrained_model stat_dict (torch state_dict)')
ap.add_argument('-dp','--data_path',required = True, type = str, help='path to data ')
ap.add_argument('-ls','--longest_sentence', required = True, type = int, help='set the limit of the sentence lenght, recommand the same to the -dt')
ap.add_argument('-wp', '--warmup', default=-1, type=float, help='portion of all training itters to warmup, -1 means not using warmup')
ap.add_argument('-t_ex_only','--train_extension_only', action='store_true', help='train only the extension module')
ap.add_argument('-not_lower','--not_lower_case', action='store_false', help='do not change case')
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

## logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(args['save_path'] + '/args.log')
logger.addHandler(file_handler)
logger.info(args)

class pre_train_BertTokenizer(BertTokenizer):
    def __init__(self, vocab_file, **kwargs):
        super(pre_train_BertTokenizer,self).__init__(vocab_file, do_lower_case=kwargs['do_lower_case'])
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

tok = pre_train_BertTokenizer(args['vocab'], do_lower_case=args['not_lower_case'])
train_dataset = Pretrain_and_Distill_Dataset(args['data_path'], tok, args['mode'], is_train=True)
val_dataset = Pretrain_and_Distill_Dataset(args['data_path'], tok, args['mode'], is_train=False)
print('done data preparation')
print('data number: '+str(len(train_dataset)))

from exBERT import BertForPreTraining, BertConfig
if args['mode'] != 'ALL':
    if args['strategy'] == 'exBERT':
        bert_config_1 = BertConfig.from_json_file(args['config'][0])
        bert_config_2 = BertConfig.from_json_file(args['config'][1])
        print("Building PyTorch model from configuration: {}".format(str(bert_config_1)))
        print("Building PyTorch model from configuration: {}".format(str(bert_config_2)))
        if args['mode'] == 'PT':
            teacher_model = BertForPreTraining(bert_config_1, bert_config_2)
        elif args['mode'] == 'DT':
            teacher_model = BertModel(bert_config_1, bert_config_2)
            # student model
            tiny_bert_config_1 = BertConfig.from_json_file(args['config_s'][0])
            tiny_bert_config_2 = BertConfig.from_json_file(args['config_s'][1])
            print("Building PyTorch model from configuration: {}".format(str(tiny_bert_config_1)))
            print("Building PyTorch model from configuration: {}".format(str(tiny_bert_config_2)))
            student_model = TinyBertForPreTraining(tiny_bert_config_1, tiny_bert_config_2)
    else:
        bert_config_1 = BertConfig.from_json_file(args['config'][0])
        print("Building PyTorch model from configuration: {}".format(str(bert_config_1)))
        if args['mode'] == 'PT':
            teacher_model = BertForPreTraining(bert_config_1)
        elif args['mode'] == 'DT':
            teacher_model = BertModel(bert_config_1)
            # student model
            tiny_bert_config_1 = BertConfig.from_json_file(args['config_s'][0])
            print("Building PyTorch model from configuration: {}".format(str(tiny_bert_config_1)))
            student_model = TinyBertForPreTraining(tiny_bert_config_1)    
else:
    # bert base teacher model
    bert_config_1 = BertConfig.from_json_file(args['config'][0])
    print("Building PyTorch model from configuration: {}".format(str(bert_config_1)))    
    teacher_model = BertModel(bert_config_1)
    # tiny exbert student model
    tiny_bert_config_1 = BertConfig.from_json_file(args['config_s'][0])
    tiny_bert_config_2 = BertConfig.from_json_file(args['config_s'][1])
    print("Building PyTorch model from configuration: {}".format(str(tiny_bert_config_1)))
    print("Building PyTorch model from configuration: {}".format(str(tiny_bert_config_2)))
    student_model = TinyBertForPreTrainingWithDistillation(tiny_bert_config_1, tiny_bert_config_2)

    tok_bert = pre_train_BertTokenizer(args['vocab_bert'], do_lower_case=args['not_lower_case'])
    train_dataset_bert = Pretrain_and_Distill_Dataset(args['data_path'], tok_bert, 'DT', is_train=True)
    val_dataset_bert = Pretrain_and_Distill_Dataset(args['data_path'], tok_bert, 'DT', is_train=False)

# num of model`s params
if args['mode'] == 'ALL' or args['mode'] == 'DT':
    teacher_total_params = sum(p.numel() for p in teacher_model.parameters())
    student_total_params = sum(p.numel() for p in student_model.parameters())
else:
    teacher_total_params = sum(p.numel() for p in teacher_model.parameters())

## load pre-trained model
if args['pretrained_model_path'] is not None:
    stat_dict = t.load(args['pretrained_model_path'], map_location='cpu')
    for key in list(stat_dict.keys()):
        if args['mode'] == 'PT':
            stat_dict[key.replace('gamma', 'weight').replace('beta', 'bias')] = stat_dict.pop(key)
        else:
            stat_dict[key.replace('bert.', '').replace('gamma', 'weight').replace('beta', 'bias')] = stat_dict.pop(key)
    # import pdb;pdb.set_trace()
    teacher_model.load_state_dict(stat_dict, strict=False) 
    
sta_name_pos = 0
if device is not 'cpu':
    if len(device_ids)>1:
        teacher_model = nn.DataParallel(teacher_model,device_ids=device_ids)
        sta_name_pos = 1
    teacher_model.to(device)

if args['strategy'] == 'exBERT' and args['mode'] == 'PT':
    if args['train_extension_only']:
        for ii,item in enumerate(teacher_model.named_parameters()):
            item[1].requires_grad=False
            if 'ADD' in item[0]:
                item[1].requires_grad = True
            if 'pool' in item[0]:
                item[1].requires_grad=True
            if item[0].split('.')[sta_name_pos]!='bert':
                item[1].requires_grad=True

if args['mode'] == 'PT':
    print('The following part of model is goinig to be trained:')
    for ii, item in enumerate(teacher_model.named_parameters()):
        if item[1].requires_grad:
            print(item[0])
    param_optimizer = list(teacher_model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
else:
    param_optimizer = list(student_model.named_parameters())
    loss_mse = MSELoss().to(device)

lr = args['learning_rate']
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
if args['mode'] == 'ALL':
    train_dataloader_bert = DataLoader(train_dataset_bert, batch_size=batch_size, shuffle=True, num_workers=2*len(args['device']))
    val_dataloader_bert = DataLoader(val_dataset_bert, batch_size=batch_size, shuffle=False, num_workers=2*len(args['device']))

save_id = 0
print_every_ndata = int(all_data_num/batch_size/20) ##output log every 0.5% of data of an epoch is processed
try:
    for epoc in range(num_epoc):
        t2 = time.time()
        train_loss = 0
        val_loss = 0
        
        if args['mode'] == 'PT':
            teacher_model.train()
        else:
            train_att_loss = 0
            train_rep_loss = 0
            student_model.train() 
            if args['mode'] != 'ALL':
                bert_iter = iter(train_dataloader_bert)
    
        for batch_ind, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
            Input_ids = batch[0].to(device)
            Token_type_ids = batch[1].to(device)
            Attention_mask = batch[2].to(device)
            Masked_lm_labels = batch[3].to(device)
            Next_sentence_label = batch[4].to(device)
            
            if args['mode'] == 'ALL':
                Input_ids2, Token_type_ids2, Attention_mask2, Masked_lm_labels2, Next_sentence_label2 = next(bert_iter)

            # import pdb;pdb.set_trace()
            optimizer.zero_grad()

            if args['mode'] == 'PT':
                loss1 = teacher_model(Input_ids,
                    token_type_ids = Token_type_ids,
                    attention_mask = Attention_mask,
                    masked_lm_labels = Masked_lm_labels,
                    next_sentence_label = Next_sentence_label)
                loss1.sum().unsqueeze(0).backward()
            elif:
                att_loss = 0.
                if args['mode'] == 'DT':
                    rep_loss = 0.
                    student_atts, student_reps = student_model(Input_ids, Token_type_ids, Attention_mask)
                    teacher_reps, teacher_atts, _ = teacher_model(Input_ids, Token_type_ids, Attention_mask)
                else:
                    ce_loss, student_atts = student_model(Input_ids, Token_type_ids, Attention_mask, Masked_lm_labels, Next_sentence_label)
                    _, teacher_atts, _ = teacher_model(Input_ids2, Token_type_ids2, Attention_mask2)
                
                # speedup 1.5x
                teacher_atts = [teacher_att.detach() for teacher_att in teacher_atts]
                
                teacher_layer_num = len(teacher_atts)
                student_layer_num = len(student_atts)
                
                assert teacher_layer_num % student_layer_num == 0
                layers_per_block = int(teacher_layer_num / student_layer_num)
                new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                    for i in range(student_layer_num)]

                for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                              student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                              teacher_att)
                    
                    att_loss += loss_mse(student_att, teacher_att)

                # rep loss
                if args['mode'] == 'DT':
                    teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]

                    new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                    new_student_reps = student_reps

                    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                        rep_loss += loss_mse(student_rep, teacher_rep)

                    loss1 = att_loss + rep_loss
                    loss1.backward()
                else:
                    loss1 = att_loss + ce_loss
                    loss1.backward()
                
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
            t.save(teacher_model.module.state_dict(),args['save_path']+'/state_dic_'+args['strategy']+'_'+str(epoc))
        else:
            t.save(teacher_model.state_dict(),args['save_path']+'/state_dic_'+args['strategy']+'_'+str(epoc))
        with open(args['save_path']+'/loss.pkl','wb') as f:
            pickle.dump([train_los_table,val_los_table,args],f)
        
        if args['mode'] == 'PT':
            teacher_model.eval()
        else:
            train_att_loss = 0
            train_rep_loss = 0
            student_model.eval() 
            if args['mode'] != 'ALL':
                bert_iter2 = iter(val_dataloader_bert)
        with t.no_grad():
            for batch_ind, batch in enumerate(val_dataloader):
                Input_ids = batch[0].to(device)
                Token_type_ids = batch[1].to(device)
                Attention_mask = batch[2].to(device)
                Masked_lm_labels = batch[3].to(device)
                Next_sentence_label = batch[4].to(device)
                
                if args['mode'] == 'ALL':
                    Input_ids2, Token_type_ids2, Attention_mask2, Masked_lm_labels2, Next_sentence_label2 = next(bert_iter2)

                if args['mode'] == 'PT':
                    loss1 = teacher_model(Input_ids,
                        token_type_ids = Token_type_ids,
                        attention_mask = Attention_mask,
                        masked_lm_labels = Masked_lm_labels,
                        next_sentence_label = Next_sentence_label)
                elif:
                    att_loss = 0.
                    if args['mode'] == 'DT':
                        rep_loss = 0.
                        student_atts, student_reps = student_model(Input_ids, Token_type_ids, Attention_mask)
                        teacher_reps, teacher_atts, _ = teacher_model(Input_ids, Token_type_ids, Attention_mask)
                    else:
                        ce_loss, student_atts = student_model(Input_ids, Token_type_ids, Attention_mask, Masked_lm_labels, Next_sentence_label)
                        _, teacher_atts, _ = teacher_model(Input_ids2, Token_type_ids2, Attention_mask2)
                    
                    # speedup 1.5x
                    teacher_atts = [teacher_att.detach() for teacher_att in teacher_atts]
                    
                    teacher_layer_num = len(teacher_atts)
                    student_layer_num = len(student_atts)
                    
                    assert teacher_layer_num % student_layer_num == 0
                    layers_per_block = int(teacher_layer_num / student_layer_num)
                    new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]

                    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                                student_att)
                        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                                teacher_att)
                        
                        att_loss += loss_mse(student_att, teacher_att)

                    # rep loss
                    if args['mode'] == 'DT':
                        teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]

                        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                        new_student_reps = student_reps

                        for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                            rep_loss += loss_mse(student_rep, teacher_rep)

                        loss1 = att_loss + rep_loss
                    else:
                        loss1 = att_loss + ce_loss

                val_log = loss1.sum().data
                val_loss+=val_log
        print('Val_loss: '+str(val_loss/(batch_ind+1)))
        if val_loss.data < best_loss:
            if len(device_ids)>1:
                t.save(teacher_model.module.state_dict(),args['save_path']+'/Best_stat_dic_'+args['strategy'])
            else:
                t.save(teacher_model.state_dict(),args['save_path']+'/Best_stat_dic_'+args['strategy'])
            best_loss = val_loss.data
            print('update!!!!!!!!!!!!')
except KeyboardInterrupt:
    print('saving stat_dict and loss table')
    with open(args['save_path']+'/kbstop_loss.pkl','wb') as f:
        pickle.dump([train_los_table,val_los_table,args],f)
    t.save(teacher_model.state_dict(),args['save_path']+'/kbstop_stat_dict')
