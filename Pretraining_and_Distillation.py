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
from torch.utils.data import DataLoader, DistributedSampler
import os
import random
from transformers.models.bert.tokenization_bert import BertTokenizer
from exBERT import BertAdam
from dataset import Pretrain_and_Distill_Dataset
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import MSELoss
# %%

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

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", required = True, type = str, help='PT or DT or ALL')
ap.add_argument("-e", "--epochs", required = True, type = int, help='number of training epochs')
ap.add_argument("-b", "--batchsize", required = True, type = int, help='training batchsize')
ap.add_argument("-sp","--save_path",required = True, type = str, help='path to storaget the loss table, stat_dict')
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
ap.add_argument('-do_lower','--do_lower_case', action='store_true', help='all text change to lower case')
ap.add_argument('--master_addr', type=str, default="127.0.0.1")
ap.add_argument('--master_port', type=str, default="22355")
ap.add_argument('--seed', type=int, default=42, help="random seed for initialization")
ap.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
args = vars(ap.parse_args())

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / world_size
    return rt

def main(rank, args):
    args['local_rank'] = rank
    if args['local_rank'] == 0:
        for ii, item in enumerate(args):
            print(item+': '+str(args[item]))
    
    ## set device
    torch.cuda.set_device(args['local_rank'])
    device = torch.device("cuda", args['local_rank'])
    n_gpu = 1
    ## initializes the distributed backend which will take care of sychronizing nodes/GPUs
    os.environ['MASTER_ADDR'] = args['master_addr']
    os.environ['MASTER_PORT'] = args['master_port']
    torch.distributed.init_process_group(backend='nccl', rank=args['local_rank'], world_size=args['world_size'])

    ## logger            
    if args['local_rank'] == 0:
        if not os.path.exists(args['save_path']):
            os.mkdir(args['save_path'])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(args['save_path'] + '/args.log')
        logger.addHandler(file_handler)
        logger.info(args)

    ## seed
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args['seed'])

    tok = pre_train_BertTokenizer(args['vocab'], do_lower_case=args['do_lower_case'])
    train_dataset = Pretrain_and_Distill_Dataset(args['data_path'], tok, args['mode'], is_train=True)
    val_dataset = Pretrain_and_Distill_Dataset(args['data_path'], tok, args['mode'], is_train=False)
    print('done data preparation')
    print('data number: '+str(len(train_dataset)))

    from exBERT.modeling import BertForPreTraining, BertConfig, BertModel, TinyBertForDistillation, TinyBertForPreTrainingWithDistillation
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
                student_model = TinyBertForDistillation(tiny_bert_config_1, tiny_bert_config_2)
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
                student_model = TinyBertForDistillation(tiny_bert_config_1)    
    else:
        # bert base teacher model
        bert_config_1 = BertConfig.from_json_file(args['config'][0])
        bert_config_2 = BertConfig.from_json_file(args['config'][1])
        print("Building PyTorch model from configuration: {}".format(str(bert_config_1)))
        print("Building PyTorch model from configuration: {}".format(str(bert_config_2)))
        teacher_model = BertForPreTraining(bert_config_1, bert_config_2)
        # tiny exbert student model
        tiny_bert_config_1 = BertConfig.from_json_file(args['config_s'][0])
        tiny_bert_config_2 = BertConfig.from_json_file(args['config_s'][1])
        print("Building PyTorch model from configuration: {}".format(str(tiny_bert_config_1)))
        print("Building PyTorch model from configuration: {}".format(str(tiny_bert_config_2)))
        student_model = TinyBertForPreTrainingWithDistillation(tiny_bert_config_1, tiny_bert_config_2)

    # num of model`s params
    if args['mode'] == 'ALL' or args['mode'] == 'DT':
        teacher_total_params = sum(p.numel() for p in teacher_model.parameters())
        student_total_params = sum(p.numel() for p in student_model.parameters())
        print(teacher_total_params)
        print(student_total_params)
        # import pdb;pdb.set_trace()
    else:
        teacher_total_params = sum(p.numel() for p in teacher_model.parameters())
        print(teacher_total_params)

    ## load pre-trained model
    if args['pretrained_model_path'] is not None:
        stat_dict = t.load(args['pretrained_model_path'], map_location='cpu')
        for key in list(stat_dict.keys()):
            if args['mode'] != 'DT':
                stat_dict[key.replace('gamma', 'weight').replace('beta', 'bias')] = stat_dict.pop(key)
            else:
                stat_dict[key.replace('bert.', '').replace('gamma', 'weight').replace('beta', 'bias')] = stat_dict.pop(key)
        # import pdb;pdb.set_trace()
        teacher_model.load_state_dict(stat_dict, strict=False)
        
    sta_name_pos = 1
    teacher_model.to(device)
    if args['mode'] == 'DT':
        teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args['local_rank']])
    else:
        if args['strategy'] != 'exBERT':
            teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args['local_rank']])
        else:
            teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args['local_rank']], find_unused_parameters=True)
    if args['mode'] != 'PT':
        student_model.to(device)
        if args['mode'] != 'ALL':
            student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[args['local_rank']], find_unused_parameters=True)
        else:
            student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[args['local_rank']])
        
    if args['strategy'] == 'exBERT' and (args['mode'] == 'PT' or args['mode'] == 'ALL'):
        if args['train_extension_only']:
            for ii,item in enumerate(teacher_model.named_parameters()):
                item[1].requires_grad=False
                if 'ADD' in item[0]:
                    item[1].requires_grad = True
                if 'pool' in item[0]:
                    item[1].requires_grad=True
                if item[0].split('.')[sta_name_pos]!='bert':
                    item[1].requires_grad=True
    
    if args['mode'] == 'PT' or args['mode'] == 'ALL':
        print('The following part of model is goinig to be trained:')
        count = 0
        for ii, item in enumerate(teacher_model.named_parameters()):
            if item[1].requires_grad:
                print(item[0])
                count += 1
        print(count)
    
        param_optimizer = list(teacher_model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    if args['mode'] != 'PT':
        param_optimizer_student = list(student_model.named_parameters())
        loss_mse = MSELoss().to(device)

    lr = args['learning_rate']
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    num_epoc = args['epochs']
    batch_size = args['batchsize']
    longest_sentence = args['longest_sentence']
    total_batch_num = int(np.ceil(len(train_dataset)/(np.trunc(batch_size/args['world_size'])*args['world_size']))) + 1
    if args['mode'] != 'DT':
        optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=lr) # warmup=args['warmup'], t_total=-1 
    if args['mode'] != 'PT':
        optimizer_grouped_parameters_student = [
        {'params': [p for n, p in param_optimizer_student if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer_student if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer_s = BertAdam(optimizer_grouped_parameters_student, lr=lr) # warmup=args['warmup'], t_total=-1 
    all_data_num = len(train_dataset)
    train_los_table = np.zeros((num_epoc,total_batch_num))
    val_los_table = np.zeros((num_epoc,total_batch_num))
    best_loss = float('inf')

    train_sampler = DistributedSampler(train_dataset, rank=args['local_rank'], num_replicas=args['world_size'])
    val_sampler = DistributedSampler(val_dataset, rank=args['local_rank'], num_replicas=args['world_size'])

    train_dataloader = DataLoader(train_dataset, batch_size=int(batch_size//args['world_size']), sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=int(batch_size//args['world_size']), sampler=val_sampler)
    
    save_id = 0
    print_every_ndata = int(all_data_num/batch_size/20) ##output log every 0.5% of data of an epoch is processed
    start_time = time.time()

    alpha = 0.

    try:
        for epoc in range(num_epoc):
            train_sampler.set_epoch(epoc)

            t2 = time.time()
            train_loss = 0
            val_loss = 0
            
            if args['mode'] == 'PT' or args['mode'] == 'ALL':
                teacher_model.train()
            if args['mode'] != 'PT':
                student_model.train() 
                
            for batch_ind, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                alpha += 1. / (total_batch_num * args['epochs'])
                
                Input_ids = batch[0].to(device)
                Token_type_ids = batch[1].to(device)
                Attention_mask = batch[2].to(device)
                Masked_lm_labels = batch[3].to(device)
                Next_sentence_label = batch[4].to(device)
                
                if args['mode'] != 'DT':
                    optimizer.zero_grad()
                if args['mode'] != 'PT':   
                    optimizer_s.zero_grad()

                if args['mode'] == 'PT':
                    loss1 = teacher_model(Input_ids,
                        token_type_ids = Token_type_ids,
                        attention_mask = Attention_mask,
                        masked_lm_labels = Masked_lm_labels,
                        next_sentence_label = Next_sentence_label,
                        mode = args['mode'])
                    loss1 = loss1.sum().unsqueeze(0)
                    loss1.backward()
                    optimizer.step()
                    
                else:
                    att_loss = 0.
                    rep_loss = 0.
                    if args['mode'] == 'DT':
                        student_reps, student_atts = student_model(Input_ids, Token_type_ids, Attention_mask)
                        teacher_reps, teacher_atts = teacher_model(Input_ids, Token_type_ids, Attention_mask, mode=args['mode'])
            
                    else:
                        total_loss_s, student_reps, student_atts = student_model(Input_ids, Token_type_ids, Attention_mask, Masked_lm_labels, Next_sentence_label)
                        total_loss, teacher_reps, teacher_atts = teacher_model(Input_ids, Token_type_ids, Attention_mask, Masked_lm_labels, Next_sentence_label, mode=args['mode'])
                    
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

                    teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]
                    
                    new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                    new_student_reps = student_reps

                    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                        rep_loss += loss_mse(student_rep, teacher_rep)
                    
                    if args['mode'] == 'DT':
                        loss1 = att_loss + rep_loss

                    else:
                        loss_t = total_loss
                        loss1 = alpha * (att_loss + rep_loss) + (1 - alpha) * total_loss_s 
                        
                        loss_t.backward()
                        optimizer.step()
                
                    loss1.backward()
                    optimizer_s.step()

                loss1 = reduce_tensor(loss1.data, dist.get_world_size())
                
                train_log = loss1.sum().data

                train_los_table[epoc,batch_ind] = train_log
                train_loss+=train_log
                if batch_ind > 0 and batch_ind % print_every_ndata == 0 and args['local_rank'] == 0: 
                    if args['mode'] != 'PT':
                        current_lr = optimizer_s.get_lr()[0]
                    else: current_lr = optimizer.get_lr()[0]
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttrain_Loss: {:.5f} \tval_Loss: {:.5f} time: {:.4f} \t lr:{:.6f}'.format(
                            epoc,
                            batch_ind*batch_size,
                            all_data_num, 100 * (batch_ind*batch_size) / all_data_num,
                            train_loss/print_every_ndata/batch_size,val_loss/print_every_ndata/batch_size,time.time()-t2 ,
                            current_lr))
                    train_loss = 0
                    val_loss = 0
                    with open(args['save_path']+'/loss.pkl','wb') as f:
                        pickle.dump([train_los_table,val_los_table,args],f)
            if args['local_rank'] == 0:
                if args['mode'] == 'PT':
                    t.save(teacher_model.module.state_dict(),args['save_path']+'/state_dic_'+args['strategy']+'_'+str(epoc))
                else:
                    t.save(student_model.module.state_dict(),args['save_path']+'/state_dic_'+args['strategy']+'_'+str(epoc))

                with open(args['save_path']+'/loss.pkl','wb') as f:
                    pickle.dump([train_los_table,val_los_table,args],f)
                
            if args['mode'] == 'PT' or args['mode'] == 'ALL':
                teacher_model.eval()
            if args['mode'] != 'PT':
                student_model.eval()
            with t.no_grad():
                for batch_ind, batch in enumerate(val_dataloader):
                    Input_ids = batch[0].to(device)
                    Token_type_ids = batch[1].to(device)
                    Attention_mask = batch[2].to(device)
                    Masked_lm_labels = batch[3].to(device)
                    Next_sentence_label = batch[4].to(device)
                
                    if args['mode'] == 'PT':
                        loss1 = teacher_model(Input_ids,
                            token_type_ids = Token_type_ids,
                            attention_mask = Attention_mask,
                            masked_lm_labels = Masked_lm_labels,
                            next_sentence_label = Next_sentence_label,
                            mode = args['mode'])
                        loss1 = loss1.sum().unsqueeze(0)
                        
                    else:
                        att_loss = 0.
                        rep_loss = 0.
                        if args['mode'] == 'DT':
                            student_reps, student_atts = student_model(Input_ids, Token_type_ids, Attention_mask)
                            teacher_reps, teacher_atts = teacher_model(Input_ids, Token_type_ids, Attention_mask, mode=args['mode'])
                        else:
                            total_loss_s, student_reps, student_atts = student_model(Input_ids, Token_type_ids, Attention_mask, Masked_lm_labels, Next_sentence_label)
                            total_loss, teacher_reps, teacher_atts = teacher_model(Input_ids, Token_type_ids, Attention_mask, Masked_lm_labels, Next_sentence_label, mode=args['mode'])
                    
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
                        
                        teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]

                        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                        new_student_reps = student_reps

                        for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                            rep_loss += loss_mse(student_rep, teacher_rep)

                        if args['mode'] == 'DT':
                            loss1 = att_loss + rep_loss
                        else:
                            loss1 = alpha * (att_loss + rep_loss) + (1 - alpha) * total_loss_s 
                        
                    loss1 = reduce_tensor(loss1.data, dist.get_world_size())
                    val_log = loss1.sum().data
                    val_loss+=val_log
            print('Val_loss: '+str(val_loss/(batch_ind+1)))
            if val_loss.data < best_loss:
                if args['local_rank'] == 0:
                    if args['mode'] == 'PT':
                        t.save(teacher_model.module.state_dict(),args['save_path']+'/Best_stat_dic_'+args['strategy'])
                    else:
                        t.save(student_model.module.state_dict(),args['save_path']+'/Best_stat_dic_'+args['strategy'])
                    best_loss = val_loss.data
                    print('update!!!!!!!!!!!!')
    except KeyboardInterrupt:
        print('saving stat_dict and loss table')
        with open(args['save_path']+'/kbstop_loss.pkl','wb') as f:
            pickle.dump([train_los_table,val_los_table,args],f)
        if args['mode'] == 'PT':
            t.save(teacher_model.state_dict(),args['save_path']+'/kbstop_stat_dict')
        else:
            t.save(student_model.state_dict(),args['save_path']+'/kbstop_stat_dict')
    end_time = time.time()
    if args['local_rank'] == 0:
        logger.info(end_time - start_time)

if __name__ == "__main__":
    args['world_size'] = torch.cuda.device_count()
    mp.spawn(main, nprocs = args['world_size'], args = (args,))
