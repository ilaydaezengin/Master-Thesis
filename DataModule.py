
from torch.utils.data import DataLoader, SequentialSampler
import functools
import torch.nn.functional as F
import torch
import pytorch_lightning as pl


class CXRDataModule(pl.LightningDataModule):
    def __init__(self,dataset): 
        super().__init__()
        self.prepare_data_per_node = True
        self.dataset = dataset
        self.prepare_data()

    def prepare_data(self):
        #dataset_df = pd.read_csv(self.csv_path)
        #dataset_df.dropna(inplace=True)
        source_texts = self.dataset.file["findings"].values
        target_texts = self.dataset.file["impressions"].values

        #source_texts = self.dataset.transform(source_texts)
        #target_texts = self.dataset.transform(target_texts)
        data = list(zip(source_texts,target_texts))
        self.data = data
        
    def __len__(self):
      return len(self.data)
      
    def __getitem__(self,idx):
      return self.data[idx]

    def process_batch(self, txt_list,tokenizer):
      source_ls = [source for source,target in txt_list]
      target_ls = [target for source,target in txt_list]

      if self.dataset.transform.__class__.__name__ == 'Switch_Sentences' or self.dataset.transform.__class__.__name__ =='kNeighbors':
        source_ls, target_ls = self.dataset.transform(source_ls,target_ls)
      else:
        source_ls = self.dataset.transform(source_ls)
        target_ls = self.dataset.transform(target_ls)
        
      source_tokens = tokenizer(source_ls,truncation=True,padding="max_length",max_length=128)
      target_tokens = tokenizer(target_ls,truncation=True,padding="max_length",max_length=128)
      input_ids = []
      attention_mask = []
      token_type_ids = []
      for i in range(len(source_tokens["input_ids"])):
        input_ids.append(source_tokens["input_ids"][i])
        input_ids.append(target_tokens["input_ids"][i])
        attention_mask.append(source_tokens["attention_mask"][i])
        attention_mask.append(target_tokens["attention_mask"][i])
        token_type_ids.append(source_tokens["token_type_ids"][i])
        token_type_ids.append(target_tokens["token_type_ids"][i])
      return torch.tensor(input_ids),torch.tensor(attention_mask),torch.tensor(token_type_ids)
          
    def train_dataloader(self,tokenizer,args):
        train_sampler = SequentialSampler(self.data)
        model_collate_fn = functools.partial(
            self.process_batch,
            tokenizer=tokenizer,
            #max_len=512
            )
        train_dataloader = DataLoader(self.data,
                                    batch_size=args.batch_size,
                                    sampler=train_sampler,
                                    #num_workers = 48,
                                    collate_fn=model_collate_fn,
                                    pin_memory=True)
        return train_dataloader