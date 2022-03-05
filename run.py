import sys
import os
import time
import torch.nn as nn
from torch import optim
from lr_scheduler import ReduceLROnPlateau
from progressbar import ProgressBar
from torch.utils.data import DataLoader
import random

from loader import *
from model import *
from config import *
from utils import *



def train():
    train_data = read_file(cfg.train_dir)
    dev_data=read_file(cfg.val_dir)

    random.seed(cfg.random_seed)
    random.shuffle(train_data)

    collator = Collator(cfg, word_to_id,cat_to_id)
    data_manager = DataLoader(train_data, collate_fn=collator, batch_size=cfg.batch_size, num_workers=0)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=cfg.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                  verbose=1, epsilon=1e-4, cooldown=0, min_lr=0, eps=1e-8)

    if cfg.mode in['FREE', 'PGD', "FGSM"]:
        delta = torch.zeros(cfg.batch_size, cfg.seq_length, cfg.embedding_size).to(cfg.device)
        delta.requires_grad = True

    #parameters in paper
    epsilon = torch.tensor(0.1)
    alpha= 10 / 255
    attack_iters=5


    best_acc = 0
    start_train_time = time.time()
    for epoch in range(1, 1 + cfg.epochs):
        print(f"Epoch {epoch}/{cfg.epochs}")
        pbar = ProgressBar(n_total=len(data_manager), desc='Training')
        model.train()
        assert model.training
        for step, batch in enumerate(data_manager):
            input_ids, label_target= batch
            input_ids = input_ids.to(cfg.device)
            label_target=label_target.to(cfg.device)

            if cfg.mode=="FGSM":
                delta.data.uniform_(-epsilon, epsilon)
                delta.data[:input_ids[0].size(0)] = clamp(delta[:input_ids[0].size(0)], -epsilon, epsilon)

                outputs = model(input_ids, delta[:input_ids[0].size(0)])
                model.zero_grad()
                loss = F.cross_entropy(outputs, label_target)
                loss.backward()
                optimizer.step()

                grad = delta.grad.detach()
                delta.data = delta + alpha * torch.sign(grad)
                delta.data[:input_ids[0].size(0)] = clamp(delta[:input_ids[0].size(0)], -epsilon, epsilon)
                delta.grad.zero_()

                outputs = model(input_ids, delta[:input_ids[0].size(0)])
                model.zero_grad()
                loss = F.cross_entropy(outputs, label_target)
                loss.backward()
                optimizer.step()
                pbar(step=step, info={'loss': loss.item()})

            elif cfg.mode=="PGD":
                delta.requires_grad = True
                for _ in range(attack_iters):
                    outputs = model(input_ids, delta[:input_ids[0].size(0)])
                    model.zero_grad()
                    loss = F.cross_entropy(outputs, label_target)
                    loss.backward()

                    grad = delta.grad.detach()
                    delta.data.uniform_(-epsilon, epsilon)
                    delta.data = delta + alpha * torch.sign(grad)
                    delta.data[:input_ids[0].size(0)] = clamp(delta[:input_ids[0].size(0)], -epsilon, epsilon)
                    delta.grad.zero_()

                delta = delta.detach()
                outputs = model(input_ids, delta[:input_ids[0].size(0)])
                model.zero_grad()
                loss = F.cross_entropy(outputs, label_target)
                loss.backward()
                optimizer.step()
                pbar(step=step, info={'loss': loss.item()})

            elif cfg.mode=="FREE":
                for _ in range(attack_iters):
                    outputs = model(input_ids, delta[:input_ids[0].size(0)])
                    model.zero_grad()
                    loss = F.cross_entropy(outputs, label_target)
                    loss.backward()

                    grad = delta.grad.detach()
                    delta.data = delta + epsilon * torch.sign(grad)
                    delta.data[:input_ids[0].size(0)] = clamp(delta[:input_ids[0].size(0)], -epsilon, epsilon)

                    delta.grad.zero_()
                    optimizer.step()
                    pbar(step=step, info={'loss': loss.item()})

            else:
                logits = model(input_ids)
                loss = F.cross_entropy(logits, label_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar(step=step, info={'loss': loss.item()})

        print("")
        acc=metric(cfg,model, dev_data, word_to_id,cat_to_id)
        logs={'acc':acc,'loss':loss.item()}
        show_info = f'Epoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        print(show_info)
        with open(cfg.history_dir,'a') as f:
            f.write(show_info+'\n')
        scheduler.epoch_step(logs['acc'], epoch)

        if logs['acc'] > best_acc:
            best_acc= logs['acc']
            if isinstance(model, nn.DataParallel):
                model_stat_dict = model.module.state_dict()
            else:
                model_stat_dict = model.state_dict()
            state = {'epoch': epoch, 'state_dict': model_stat_dict}
            torch.save(state, cfg.save_path)
    train_time =time.time()
    print('Total train time: %.4f minutes', (train_time - start_train_time) / 60)


def test():

    test_data=read_file(cfg.test_dir)
    states = torch.load(cfg.save_path)
    state = states['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)

    accuracy= metric(cfg,model,test_data,word_to_id, cat_to_id,mode='test')
    print(f'{accuracy}')


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_rnn.py [train / test]""")

    cfg=TextConfig()
    filenames = [cfg.train_dir, cfg.test_dir, cfg.val_dir]
    if not os.path.exists(cfg.vocab_dir):
        build_vocab(filenames, cfg.vocab_dir, cfg.vocab_size)
    word_to_id=read_vocab(cfg.vocab_dir)
    cat_to_id=read_category()

    model = CNNModel(cfg)
    model.to(cfg.device)

    if sys.argv[1] == 'train':
        train()
    else:
        test()