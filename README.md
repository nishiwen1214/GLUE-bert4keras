# 基于bert4keras的GLUE基准代码
GLUE benchmark: [General Language Understanding
Evaluation](https://gluebenchmark.com/)

### 实验结果：

| Task  | Metric                       | Huggingface | Our (bert4keras)|
|-------|------------------------------|-------------|---------------|
| CoLA  | Matthews corr                | 56.53       | 60.07         |
| SST-2 | Accuracy                     | 92.32       |          |
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
