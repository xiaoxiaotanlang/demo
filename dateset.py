import os
from torch.utils.data import Dataset
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

# ------------------------------------------------------------------------------------------------------------------------

class NerDataset(Dataset):

    def __init__(self, sents):
        self.sents = sents

    def __getitem__(self, index):
        return self.sents[index]

    def __len__(self):
        return len(self.sents)

def generate_dataloader(dataset,tokenizer,batch_size):
    """
    创建模型训练用dataloader并返回
    """
    def collate_fn(batch):
        """
        批次数据转换为模型训练用张量
        """
        batch_data = tokenizer(batch, padding='max_length', max_length = 27, truncation=True,return_tensors='pt')           # 填充到固定长度27，不管彼此是否为1

        return batch_data
        
    return DataLoader(dataset=dataset,batch_size=batch_size,collate_fn=collate_fn)


if __name__ == '__main__':
    from read_corpus import corpus
    # 加载语料
    corpus_dir = os.path.join(os.path.dirname(__file__), 'vocab_vector.txt')
    gkh,cpcn = corpus(corpus_dir)
    dataset = NerDataset(cpcn)
    # 加载tokenizer分词器
    tokenizer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../zhuanli_ner/bert_model/bert-base-chinese'))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # 测试dataloader
    dataloader = generate_dataloader(dataset, tokenizer, batch_size=8)
    for data in dataloader:
        print(data)
        break