# fast_adversarial_for_text_classification
本项目基于TextCNN，测试三种对抗训练模型（FGSM，PGD，FREE）在text classification上的表现。主要参考论文[<Fast is better than free: Revisiting adversarial training>](https://arxiv.org/pdf/2001.03994.pdf)涉及的三个对抗训练方法：FGSM（Fast Gradient Sign Method）、PGD（projected gradient decent）、FREE（Free adversarial based on FGSM）。这三种方法主要差异在于delta参数的初始化和更新方式上，其差异性可以见下面三个模型对应的伪代码。<br><br>

 
**FGSM模型的计算逻辑:**<br><br>
![image](https://github.com/cjymz886/fast_adversarial_for_text_classification/blob/main/imgs/PFGM.png)<br><br>
**PGD,FREE模型的计算逻辑:**<br><br>
![image](https://github.com/cjymz886/fast_adversarial_for_text_classification/blob/main/imgs/pgd.png)<br><br>

1 环境
=
python3.7<br>
torch 1.8.0+cu111<br>
scikit-learn<br>
scipy<br>
numpy<br>

3 数据集
=
本实验同样是使用THUCNews的一个子集进行训练与测试，数据集请自行到[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)下载，请遵循数据提供方的开源协议;<br><br>
文本类别涉及10个类别：categories = \['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']；<br><br>
cnews.train.txt: 训练集(5000*10)<br>
cnews.val.txt: 验证集(500*10)<br>
cnews.test.txt: 测试集(1000*10)<br><br>
训练所用的数据，以及训练好的词向量可以下载：链接: [https://pan.baidu.com/s/1DOgxlY42roBpOKAMKPPKWA](https://pan.baidu.com/s/1DOgxlY42roBpOKAMKPPKWA)，密码: up9d<br><br>
