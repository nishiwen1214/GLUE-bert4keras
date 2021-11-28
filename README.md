# 基于bert4keras的GLUE基准代码

引用[苏神博客](https://kexue.fm/archives/8739)中的话：“事实上，不管是tensorflow还是pytorch，不管是CLUE还是GLUE，笔者认为能找到的baseline代码，都很难称得上人性化，试图去理解它们是一件相当痛苦的事情。” (附上苏神的中文[CLUE基准代码](https://github.com/bojone/CLUE-bert4keras))

本人也是感同身受，既然有了中文CLUE基准代码，那么英文GLUE基准代码也得搞一个，所以决定基于bert4keras实现一套GLUE的baseline。经过测试，基本上复现了Huggingface的基准成绩，并且大多数任务还更优。最重要的是，所有代码尽量保持了清晰易读的特点（bert4keras的最大特点）。

GLUE benchmark: [General Language Understanding
Evaluation](https://gluebenchmark.com/)

### 欢迎star和提问～

### 实验结果：
- val set:

| Task  | Metric                       | [BERT-base (Huggingface)](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification?fbclid=IwAR0Y4_Eer7ovaEJpRMpE1S91AsrOdEK97-iah6mupW9RATs2XMPVzQZCNz8) | [BERT-base (Our)](https://github.com/nishiwen1214/GLUE-bert4keras)|
|-------|------------------------------|-------------|---------------|
| CoLA  | Matthews corr                | 56.53       | 61.53         |
| SST-2 | Accuracy                     | 92.32       | 92.66         |
| MRPC  | F1/Accuracy                  | 88.85/84.07 | 88.53/84.17   |
| STS-B | Pearson/Spearman corr.       | 88.64/88.48 | 89.20/88.77   |
| QQP   | Accuracy/F1                  | 90.71/87.49 | 90.81/87.66   |
| MNLI  | Matched acc./Mismatched acc. | 83.91/84.10 | 84.02/84.24   |
| QNLI  | Accuracy                     | 90.66       | 91.42         |
| RTE   | Accuracy                     | 65.70       | 69.68         |
| WNLI  | Accuracy                     | 56.34       | 56.34         |

- test set:
<img width="1367" alt="image" src="https://user-images.githubusercontent.com/56249874/143742252-4a2d69ba-2f6a-4989-a5f8-f69e1db56691.png">
<img width="924" alt="image" src="https://user-images.githubusercontent.com/56249874/143735793-762babad-f43b-482e-87b5-61210720a34f.png">

### 使用
- 下载[GLUE数据集](https://gluebenchmark.com/)和bert预训练的权重(这里使用的是[Google原版bert](https://github.com/google-research/bert))到指定文件夹；
- 例如：训练CoLA，直接运行 `python CoLA.py`。

### 环境
- 软件：bert4keras>=0.10.8, tensorflow = 1.15.0, keras = 2.3.1；
- 硬件：结果是用一张RTX 2080（12G）跑出来的。

### 更新
- 2021.11.28，增加code的test set的预测功能，并且上传到GLUE网站进行评估，结果已公开。
