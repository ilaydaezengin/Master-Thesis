from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import faiss
import stanza
import torch


from transformers import BertForMaskedLM, BertConfig, BertTokenizer


class Sentence_Tokenizer:
    def __init__(self):
        stanza.download('en')
        self.stanza_tokenizer = stanza.Pipeline('en', processors='tokenize', use_gpu =False)

        
    def __call__(self, report):
            return self.process(report)
        

    def process(self, r_section):
        doc = self.stanza_tokenizer(r_section)

        sentences = [sentence.text for sentence in doc.sentences]

        return sentences
    


class SectionsDataset(Dataset):
   
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the text data.
            transform (callable, optional): Optional transform to be applied
                on a sample (augmentations).
        """
        self.file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        
        findings = self.file.iloc[idx, 0]
        impressions = self.file.iloc[idx,1]
        sample = {'findings': findings, 'impressions': impressions}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
#remove sentence
class Drop(object):
    def __init__(self,sentence_tokenizer, nr):
        self.sentence_tokenizer = sentence_tokenizer
        self.nr = nr

    def __call__(self, arr):
        arr2 = []
        for f in range(len(arr)):
            f_splitted = self.sentence_tokenizer.process(arr[f])
            for i in range(self.nr):
                if self.nr < len(f_splitted):
                    n_f = np.random.randint(0, len(f_splitted))
                    f_splitted.pop(n_f)
            arr2.append(' '.join(str(x) for x in f_splitted))
                
        return arr2

 
#swap sentences between sections
class Switch_Sentences(object):
    def __init__(self,sentence_tokenizer):
        self.sentence_tokenizer = sentence_tokenizer

    def __call__(self, f_arr, i_arr):
        arr_f = []
        arr_i = []
        for idx in range(len(f_arr)):

            f_splitted = self.sentence_tokenizer.process(f_arr[idx])
            i_splitted = self.sentence_tokenizer.process(i_arr[idx])
   
            n_f = np.random.randint(0, len(f_splitted))
            sent_f = f_splitted[n_f]
            f_splitted.pop(n_f)
            n_i = np.random.randint(0, len(i_splitted))
            sent_i = i_splitted[n_i]
            i_splitted.pop(n_i)
            f_splitted.append(sent_i)
            i_splitted.append(sent_f)

            arr_f.append(' '.join(str(x) for x in f_splitted))
            arr_i.append(' '.join(str(x) for x in i_splitted))
            
        return arr_f, arr_i  
    
class Shuffle(object):
    def __init__(self,sentence_tokenizer):
        self.sentence_tokenizer = sentence_tokenizer

    def __call__(self, arr):
        arr2 = []
        for f in range(len(arr)):
            f_splitted = self.sentence_tokenizer.process(arr[f])
            np.random.shuffle(f_splitted)
            #f_shuffled = " ".join(str(x) for x in f_splitted)
            #arr2.append(f_shuffled)
            arr2.append(' '.join(f_splitted))

        return arr2
    
class kNeighbors(object):
    def __init__(self, sentence_tokenizer, pickle_file_f, pickle_file_i):
        self.sentence_tokenizer = sentence_tokenizer

        import pickle
        with open(pickle_file_f, 'rb') as f:
            data = pickle.load(f)
        with open(pickle_file_i, 'rb') as i:
            data2 = pickle.load(i)
        self.vectors_f = data
        self.vectors_i = data2
    

        rand_idx_f = np.random.randint(0, len(self.vectors_f.sentences), (1, int(0.01 * len(self.vectors_f.sentences))))
        sub_dt_f = {'sentences': self.vectors_f.sentences[rand_idx_f][0], 'embeds': self.vectors_f.embeds[rand_idx_f][0]}
        self.db_vectors_f = pd.Series(sub_dt_f)
        self.index_f = self.create_index(self.db_vectors_f)

        rand_idx_i = np.random.randint(0, len(self.vectors_i.sentences), (1, int(0.01 * len(self.vectors_i.sentences))))
        sub_dt_i = {'sentences': self.vectors_i.sentences[rand_idx_i][0], 'embeds': self.vectors_i.embeds[rand_idx_i][0]}
        self.db_vectors_i = pd.Series(sub_dt_i)
        self.index_i = self.create_index(self.db_vectors_i)

    def create_index(self, db_vectors):
        db_vectors_in = db_vectors.embeds[:, 0].astype('float32')
        quantizer = faiss.IndexFlatIP(db_vectors.embeds[:, 0].shape[1]) 
        index = faiss.IndexIVFFlat(quantizer, db_vectors.embeds[:, 0].shape[1], 100, faiss.METRIC_INNER_PRODUCT)
        index.train(db_vectors_in)
        index.add(db_vectors_in)
        index.nprobe = 100
        return index  
    
    def return_index(self, index, query):
        k = 2
        dist, idx = index.search(query, k)
        return idx[0][1]

    def __call__(self, f_arr, i_arr):
        #rand_idx_f = np.random.randint(0, len(self.vectors_f.sentences), (1, int(0.1 * len(self.vectors_f.sentences))))
        #sub_dt = {'sentences': self.vectors_f.sentences[rand_idx_f][0], 'embeds': self.vectors_f.embeds[rand_idx_f][0]}
        arr_f = []
        arr_i = []
        for idx in range(len(f_arr)):
            f_arr_mid = []
            f_splitted = self.sentence_tokenizer.process(f_arr[idx])
            for sent in f_splitted:
                #tokens = tokenizer_1(sent,truncation=True,padding=\"max_length\",max_length=25)
                #sent_outputs_f = model_1(torch.tensor(tokens.input_ids).unsqueeze(0), torch.tensor(tokens.attention_mask).unsqueeze(0), torch.tensor(tokens.token_type_ids).unsqueeze(0))[0][:,0,:].reshape(1,-1).detach().numpy().astype('float32')
                idx_sen = self.vectors_f.sentences.tolist().index(sent)
                sent_outputs_f = self.vectors_f.embeds[idx_sen]
                
                idx_f = self.return_index(self.index_f, sent_outputs_f)
                f_arr_mid.append(self.db_vectors_f.sentences[idx_f])
            arr_f.append(' '.join(f_arr_mid))


            i_arr_mid = []
            i_splitted = self.sentence_tokenizer.process(i_arr[idx])
            for sent in i_splitted:
                #tokens = tokenizer_1(sent,truncation=True,padding=\"max_length\",max_length=25)
                 #sent_outputs_i = model_1(torch.tensor(tokens.input_ids).unsqueeze(0), torch.tensor(tokens.attention_mask).unsqueeze(0), torch.tensor(tokens.token_type_ids).unsqueeze(0))[0][:,0,:].reshape(1,-1).detach().numpy().astype('float32')
                idx_sen = self.vectors_i.sentences.tolist().index(sent)
                sent_outputs_i = self.vectors_i.embeds[idx_sen]
                
                idx_i = self.return_index(self.index_i, sent_outputs_i)
                i_arr_mid.append(self.db_vectors_i.sentences[idx_i])
            arr_i.append(' '.join(i_arr_mid))

        return arr_f, arr_i