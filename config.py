import torch

class TextConfig():
    embedding_size = 100
    vocab_size = 5800

    seq_length = 300
    num_classes = 10

    kernel_dim=256
    kernel_sizes=[2,3,4]
    random_seed = 2009
    hidden_size=128

    keep_prob = 0.5
    lr = 1e-3
    batch_size = 100    #选择不丢失数据，保证为训练集(5万条)整除
    epochs = 20

    device = torch.device("cuda")

    mode='FREE'   #FGSM,FREE,PGD default

    train_dir = './data/cnews.train.txt'
    test_dir = './data/cnews.test.txt'
    val_dir = './data/cnews.val.txt'
    vocab_dir = './data/vocab.txt'

    save_path='./save_models/'+mode+'/best_model.weights'
    history_dir='./save_models/'+mode+'/history.txt'