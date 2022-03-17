Efficient Domain Extension Distillation(EDx2)

1. Pretraining

In command line:

    python Pretraining.py 
      -m PT
      -e 3 
      -b 256 
      -sp (path dir to save outputs)
      -dv (gpu devices to use) -lr 1e-04 
      -str exBERT
      -config (path to config file of original BERT) (path to config file of extension module)  
      -vocab ./config_and_vocab/exBERT/exBERT_vocab.txt 
      -pm_p (path to state dict)
      -dp (path to your training data)
      -ls (max length of sequence)

2. Distillation
    

3. EDx2 (Pretraining + Distillation)
