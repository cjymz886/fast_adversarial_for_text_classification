import codecs
import re
import torch
import numpy as np
from collections import  Counter
from collections import namedtuple



def read_file(file_dir):

    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)")  # 去掉标点符号和数字类型的字符

    SentInst = namedtuple('SentInst', 'tokens label')
    data =[]
    with codecs.open(file_dir ,'r' ,encoding='utf-8') as f:
        for line in f:
            try:
                line = line.split('\t')
                assert len(line) == 2
                label ,text =line[0],line[1]
                content =[]
                for w in text:
                    if re_han.match(w):
                        content.append(w)
                sent_inst = SentInst(content, label)
                data.append(sent_inst)
            except:
                pass
    return data


def build_vocab(file_dirs,vocab_dir,vocab_size=6000):
    """
    利用训练集和测试集的数据生成字级的词表
    """
    all_data = []
    for filename in file_dirs:
        for line in read_file(filename):
            content= line.tokens
            all_data.extend(content)
    counter=Counter(all_data)
    count_pairs=counter.most_common(vocab_size-1)
    words,_=list(zip(*count_pairs))
    words=['<PAD>']+list(words)

    with codecs.open(vocab_dir,'w',encoding='utf-8') as f:
        f.write('\n'.join(words)+'\n')

def read_vocab(vocab_dir):
    words=codecs.open(vocab_dir,'r',encoding='utf-8').read().strip().split('\n')
    word_to_id=dict(zip(words,range(len(words))))
    return word_to_id


def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id=dict(zip(categories,range(len(categories))))
    return cat_to_id


def seq_padding(seq,seq_len):
    if len(seq)>seq_len:
        seq=seq[:seq_len]
    while len(seq)<seq_len:
        seq.append(0)
    return seq


class data_generator:
    def __init__(self, cfg, data, word_to_id, cat_to_id):
        self.cfg = cfg
        self.data = data
        self.steps = len(self.data) // self.cfg.batch_size
        if len(self.data) % self.cfg.batch_size != 0:
            self.steps += 1
        self.word_to_id = word_to_id
        self.cat_to_id = cat_to_id

    def __len__(self):
        return self.steps

    def __iter__(self):
        idxs = list(range(len(self.data)))
        np.random.seed(self.cfg.random_seed)
        np.random.shuffle(idxs)

        text_batch, label_batch= [], []
        for idx in idxs:
            line = self.data[idx]
            content = line.tokens
            label = line.label

            input_ids = [self.word_to_id[x] if x in self.word_to_id else 0 for x in content]
            label_ids = self.cat_to_id[label]
            input_ids=seq_padding(input_ids,self.cfg.seq_length)

            text_batch.append(input_ids)
            label_batch.append(label_ids)


            if len(text_batch) == self.cfg.batch_size or idx == idxs[-1]:
                text_batch = torch.LongTensor(text_batch)
                label_batch=torch.LongTensor(label_batch)

                yield text_batch, label_batch
                text_batch, label_batch = [], []





class Collator(object):
    def __init__(self, cfg, word_to_id,cat_to_id):
        # super(Collator,self).__init__()
        self.cfg = cfg
        self.word_to_id = word_to_id
        self.cat_to_id = cat_to_id

    def __call__(self, batch):
        batch_token, batch_label = [], []
        for line in batch:
            content = line.tokens
            label = line.label

            input_ids = [self.word_to_id[x] if x in self.word_to_id else 0 for x in content]
            label_ids = self.cat_to_id[label]
            input_ids=seq_padding(input_ids,self.cfg.seq_length)

            batch_token.append(input_ids)
            batch_label.append(label_ids)

        batch_token = torch.LongTensor(batch_token)
        batch_label = torch.LongTensor(batch_label)
        return batch_token, batch_label