import torch
import torch.nn as nn
import torch.nn.functional as F



class CNNModel(nn.Module):
    def __init__(self, cfg):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_size)
        self.convs = nn.ModuleList([nn.Conv2d(1,cfg.kernel_dim, (K, cfg.embedding_size)) for K in cfg.kernel_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3 * cfg.kernel_dim, cfg.hidden_size)
        self.fc2=nn.Linear(cfg.hidden_size, cfg.num_classes)


    def forward(self, inputs_ids,attack=None,is_training=True):
        embs = self.embedding(inputs_ids)
        if attack is not None:
            embs=embs+attack        #加入干扰信息
        embs=embs.unsqueeze(1)

        inputs = [F.relu(conv(embs)).squeeze(3) for conv in self.convs]
        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]
        concated = torch.cat(inputs, 1)

        if is_training:
            concated = self.dropout(concated)
        fc = self.fc1(concated)
        out = self.fc2(fc)
        return out

