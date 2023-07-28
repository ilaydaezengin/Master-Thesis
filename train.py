import data
import CXRBERT
import DataModule
import EmbeddingModule
import pandas as pd
import torch
import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

s = data.Sentence_Tokenizer()
checkpoint = 'allenai/scibert_scivocab_cased'
config = CXRBERT.CXRBertConfig.from_pretrained(checkpoint)
tokenizer = CXRBERT.CXRBertTokenizer.from_pretrained(checkpoint,padding="max_length", truncation=True, max_length=256)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
name = torch.cuda.get_device_name(0)
n_cpu = os.cpu_count()

class Arguments():
  def __init__(self):
    #self.model_name_or_path = 'bert-base-uncased'
    #self.max_seq_length = 32
    self.lr = 2e-5 
    self.warmup_proportion = 0.03 
    self.weight_decay = 0.01  
    self.num_train_epochs = 50
    self.gradient_accumulation_steps = 1
    self.pad_to_max_length = True
    self.batch_size = 128 
    self.output_dir = 'model_outputs'

args = Arguments()

#sec_dataset_with_neighbors = SectionsDataset(csv_file='./finds_imps_half_final.csv', root_dir='./', transform = kNeighbors(s,'./f_sent_embeds_dict.pkl', './i_sent_embeds_dict.pkl') )
sec_dataset_with_drop = data.SectionsDataset(csv_file='./finds_imps_half_final.csv', root_dir='./', transform = data.Drop(s, 1))
sec_dataset_with_shuffle = data.SectionsDataset(csv_file='./finds_imps_half_final.csv', root_dir='./', transform = data.Shuffle(s))
sec_dataset_with_switch = data.SectionsDataset(csv_file='./finds_imps_half_final.csv', root_dir='./', transform = data.Switch_Sentences(s))
sec_dataset_with_neighbors2 = data.SectionsDataset(csv_file='./finds_imps_half_final.csv', root_dir='./', transform = data.kNeighbors(s,'./f_sent_embeds_dict.pkl', './i_sent_embeds_dict.pkl') ) 

dt = sec_dataset_with_neighbors2




model = EmbeddingModule(args, config, checkpoint, dt)
data_module = DataModule.CXRDataModule(dt)
logger = WandbLogger(project="CXR-BERT", log_model="all")

model_checkpoint = ModelCheckpoint()
lr_monitor = LearningRateMonitor(logging_interval='step')


#no need to put optimizer and scheduler, trainer object will update for each epoch
trainer = Trainer(
    accelerator="gpu", 
    logger=logger,
    max_epochs=args.num_train_epochs,
    log_every_n_steps=1,
    callbacks=[model_checkpoint, lr_monitor]
)


trainer.fit(model, data_module)