# fast_adversarial_for_text_classification
本项目基于TextCNN，测试三种对抗训练模型（FGSM，PGD，FREE）在text classification上的表现。主要参考论文[<Fast is better than free: Revisiting adversarial training>](https://arxiv.org/pdf/2001.03994.pdf)涉及的三个对抗训练方法：FGSM（Fast Gradient Sign Method）、PGD（projected gradient decent）、FREE（Free adversarial based on FGSM）。这三种方法主要差异在于delta参数的初始化和更新方式上，其差异性可以见下面三个模型对应的伪代码。
