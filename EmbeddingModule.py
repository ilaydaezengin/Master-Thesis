import pytorch_lightning as pl
import CXRBERT
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


class EmbeddingModule(pl.LightningModule):
    def __init__(self, args, config, checkpoint, dt):
        super().__init__()
        self.args = args
        self.model = CXRBERT.CXRBertModel(config).from_pretrained(checkpoint)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        #self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.temperature = 0.5

    def forward(self, X,y,z):
        return self.model(X,y,z)

    def _compute_losses(self, f_embeddings, i_embeddings):
        logits = (i_embeddings @ f_embeddings.T) / self.temperature
        f_similarity = f_embeddings @ f_embeddings.T
        i_similarity = i_embeddings @ i_embeddings.T
        targets = F.softmax(
            (f_similarity + i_similarity) / 2 * self.temperature, dim=-1
        )
        f_loss = (-targets.T * self.log_softmax(logits.T)).sum(1)
        i_loss = (-targets * self.log_softmax(logits)).sum(1)
        return (f_loss + i_loss) / 2.0

    


    def step(self, batch, step_name = "train"):
        input_ids,attention_mask,token_type_ids = batch
        #outputs = self.forward(input_ids,attention_mask,token_type_ids)
        a = int(len(input_ids) / 2)
        f_cls_embeddings = self.model.get_projected_text_embeddings(input_ids[:a],attention_mask[:a])
        i_cls_embeddings = self.model.get_projected_text_embeddings(input_ids[a:],attention_mask[a:])
        loss = self._compute_losses(f_cls_embeddings,i_cls_embeddings).mean()
        
        #loss = self.info_nce_loss(f_cls_embeddings,i_cls_embeddings)
        train_loss = self.all_gather(loss)
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}
        self.log('train_loss', loss)
        
        return { ("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
                        "progress_bar": {loss_key: loss}}
    
    def training_step(self, batch, batch_idx):
        
        return self.step(batch, "train")


    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias','LayerNorm.weight']
        param_groups = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        num_train_optimization_steps = int(len(self.dt) / self.args.batch_size / self.args.gradient_accumulation_steps) * self.args.num_train_epochs
        warmup_steps = int(self.args.warmup_proportion * num_train_optimization_steps)

        optimizer = AdamW(param_groups,lr=self.args.lr)

        scheduler = get_linear_schedule_with_warmup(
         optimizer,
         num_warmup_steps=warmup_steps,
         num_training_steps=num_train_optimization_steps)
        #scheduler = self.warmup_linear_schedule(optimizer, warmup_steps, num_train_optimization_steps)
        return [optimizer], [scheduler]