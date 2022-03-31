# Efficient Domain Extension Distillation(EDx2)

## 1. Pretraining

#### case 1 (exBERT)

In command line:

    CUDA_VISIBLE_DEVICES=0,1 python Pretraining_and_Distillation.py 
      -m PT
      -e 3 
      -b 256 
      -sp (path dir to save outputs)
      -lr 1e-04 
      -str exBERT
      -config (path to config file of original BERT) (path to config file of extension module)  
      -vocab (path to vocab file) 
      -pm_p (path to state dict)
      -dp (path to your training data)
      -ls (max length of sequence)
      -do_lower
      -t_ex_only

#### case 2 (ex: bioBERT, sciBERT, etc.)

In command line:

    CUDA_VISIBLE_DEVICES=0,1 python Pretraining_and_Distillation.py 
      -m PT
      -e 3 
      -b 256 
      -sp (path dir to save outputs)
      -lr 1e-04 
      -str bioBERT
      -config (path to config file of original BERT)
      -vocab (path to vocab file) 
      -pm_p (path to state dict)
      -dp (path to your training data)
      -ls (max length of sequence)

## 2. Distillation

In command line:

    CUDA_VISIBLE_DEVICES=0,1 python Pretraining.py 
      -m DT
      -e 3
      -b 128 
      -sp (path dir to save outputs)
      -lr 1e-04 
      -str exBERT
      -config (path to config file of teacher model)
      -config_s (path to config file of student model) 
      -vocab (path to vocab file) 
      -pm_p (path to state dict of pretrained teacher model)
      -dp (path to your training data)
      -ls (max length of sequence)
      -do_lower


## 3. EDx2 (Pretraining + Distillation)

In command line:

    CUDA_VISIBLE_DEVICES=0,1 python Pretraining.py 
      -m ALL
      -e 3 
      -b 128 
      -sp (path dir to save outputs)
      -lr 1e-04 
      -str exBERT
      -config (path to config file of original BERT)
      -config_s (path to config file of TinyBERT) (path to config file of tiny extension module)  
      -vocab (path to vocab file)
      -pm_p (path to state dict of pretrained BERT)
      -dp (path to your training data)
      -ls (max length of sequence)
      -do_lower
