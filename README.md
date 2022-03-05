# fast_adversarial_for_text_classification
本项目基于TextCNN，测试三种对抗训练模型（FGSM，PGD，FREE）在text classification上的表现。主要参考论文[<Fast is better than free: Revisiting adversarial training>](https://arxiv.org/pdf/2001.03994.pdf)涉及的三个对抗训练方法：FGSM（Fast Gradient Sign Method）、PGD（projected gradient decent）、FREE（Free adversarial based on FGSM）。这三种方法主要差异在于delta、alpha参数的初始化和更新方式上，其差异性可以见下面三个模型对应的伪代码。<br>

 
**FGSM模型的计算逻辑:**<br>
![image](https://github.com/cjymz886/fast_adversarial_for_text_classification/blob/main/imgs/PFGM.png)<br>
**PGD,FREE模型的计算逻辑:**<br>
![image](https://github.com/cjymz886/fast_adversarial_for_text_classification/blob/main/imgs/pgd.png)<br>

1 环境
=
python3.7<br>
torch 1.8.0+cu111<br>
scikit-learn<br>
scipy<br>
numpy<br>

2 数据集
=
本实验同样是使用THUCNews的一个子集进行训练与测试，数据集请自行到[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)下载，请遵循数据提供方的开源协议;<br>
文本类别涉及10个类别：categories = \['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']；<br>
cnews.train.txt: 训练集(5000*10)<br>
cnews.val.txt: 验证集(500*10)<br>
cnews.test.txt: 测试集(1000*10)<br>
训练所用的数据，以及训练好的词向量可以下载：链接: [https://pan.baidu.com/s/1DOgxlY42roBpOKAMKPPKWA](https://pan.baidu.com/s/1DOgxlY42roBpOKAMKPPKWA)，密码: up9d<br>
 
3 生成对抗样本思路
=
本实验在文本Embedding的上，利用对抗训练方法产生attack后，然后再加入embedding中，最后利用cnn来进行文本特征学习。其实现部分代码如下：
 ```
     def forward(self, inputs_ids,attack=None,is_training=True):
        embs = self.embedding(inputs_ids)
        if attack is not None:
            embs=embs+attack        #加入干扰信息
        embs=embs.unsqueeze(1)
        ....
        out = self.fc2(fc)
        return out
 ```

4 运行步骤
=
首先在config.py中选择要运行的mode，训练与测试分别执行如下：<br>
python run.py train <br>
python run.py test <br>

 
5 训练结果
=
四种模型在测试集上实验结果如下：
| **Model** | **Accuracy** | **Precision**|**Recall** |**F1-score**|
| ------| ------| ------| ------| ------|
|TextCNN|	95.14	|95.16|	95.14	|95.11|
|FGSM|	95.53	|95.60	|95.53|	95.50|
|PGD	|95.63|	95.67|	95.63|95.60|
|FREE	|95.49	|95.54|	94.49	|95.46|

四种模型训练消耗的时间(minutes)对比如下：
|**model**| **total_cost**|	**mean_cost**|
| ------| ------| ------| 
|TextCNN|	3.7	|0.185|
|FGSM|	5.83|	0.2915|
|PGD|	13.54	|0.677|
|FREE	|12.22|	0.611|

6 结论
=
通过本次实验，有以下几点结论与想法：<br>
 + 对抗训练训练方法的确有助于提高文本分类任务效果；<br>
 + FGSM方法虽然提高了训练效率，但并不影响推理速度，而且NLP领域任务都不用大的轮数，所以PGD方法更合适些；<br>
 + 三种方法涉及delta、alpha超参数的初始化设定，面临不同的任务，会有变动，增加探寻合适参数的难度；<br>
 + 在文本分类中，觉得更好用word2vec或者bert方式初始化向量来进行干扰样本生成，应比随机初始化embedding方式更合适，而且可以根据高频率词的分布来初始化delta、alpha更合理；<br>
 + 若在本论文的FGSM基础，考虑如何更稳定或自动化的方式初始化delta等参数仍值得去研究。<br>

 # Reference
1.[FAST IS BETTER THAN FREE: REVISITING ADVERSARIAL TRAINING](https://arxiv.org/pdf/2001.03994.pdf)<br>
2.[https://github.com/locuslab/fast_adversarial](https://github.com/locuslab/fast_adversarial)<br>
3.[Adversarial Training Methods for Semi-Supervised Text Classification.](https://arxiv.org/pdf/1605.07725.pdf)<br>
4.[[论文笔记] Projected Gradient Descent (PGD)](https://zhuanlan.zhihu.com/p/345228508)
 
