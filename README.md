# 基于bert4keras的GLUE基准代码

引用[苏神博客](https://kexue.fm/archives/8739)中的话：“事实上，不管是tensorflow还是pytorch，不管是CLUE还是GLUE，笔者认为能找到的baseline代码，都很难称得上人性化，试图去理解它们是一件相当痛苦的事情。” (附上苏神的中文[CLUE基准代码](https://github.com/bojone/CLUE-bert4keras))

本人也是感同身受，既然有了中文的CLUE基准代码，那么所以英文的GLUE基准代码也得搞一个，所以决定基于bert4keras实现一套GLUE的baseline。经过测试，基本上复现了Huggingface的基准成绩，并且可以说有些任务还更优。最重要的是，所有代码尽量保持了清晰易读的特点（bert4keras的最大特点）。


GLUE benchmark: [General Language Understanding
Evaluation](https://gluebenchmark.com/)

### 实验结果：

| Task  | Metric                       | Huggingface | Our (bert4keras)|
|-------|------------------------------|-------------|---------------|
| CoLA  | Matthews corr                | 56.53       | 60.07         |
| SST-2 | Accuracy                     | 92.32       | 92.89         |
| MRPC  | F1/Accuracy                  | 88.85/84.07 |           |
| STS-B | Pearson/Spearman corr.       | 88.64/88.48 |           |
| QQP   | Accuracy/F1                  | 90.71/87.49 |       |
| MNLI  | Matched acc./Mismatched acc. | 83.91/84.10 |        |
| QNLI  | Accuracy                     | 90.66       |         |
| RTE   | Accuracy                     | 65.70       |             |
| WNLI  | Accuracy                     | 56.34       |            |

### 使用
- 下载GLUE数据集和bert预训练的权重(这里使用的是谷歌的预训练权重)到指定文件夹；
- 例如：训练CoLA，直接 python CoLA.py

### 环境
- 软件：bert4keras>=0.10.8, tensorflow = 1.15.0, keras = 2.3.1
- 硬件：结果是用一张RTX 2080（12G）跑出来的。