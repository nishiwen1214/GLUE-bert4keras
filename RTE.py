#! -*- coding:utf-8 -*-
# https://github.com/nishiwen1214/GLUE-bert4keras
# 句子对分类任务，RET数据集
# val_acc: 69.675

import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense
from tqdm import tqdm
import csv
import os
# 选择使用第几张GPU卡，'0'为第一张
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

set_gelu('tanh')  # 切换gelu版本

labels = ['entailment', 'not_entailment']
num_classes = len(labels)
maxlen = 128
batch_size = 32
config_path = './uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './uncased_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
    i = 1
    with open(filename, encoding='utf-8') as f:
        for l in f:
            if i == 1: # 跳过数据第一行
                i = 2
            else:
                _, text1, text2, label = l.strip().split('\t')
                D.append((text1, text2, labels.index(label)))
    return D

def load_data_test(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
    i = 1
    with open(filename, encoding='utf-8') as f:
        for l in f:
            if i == 1: # 跳过数据第一行
                i = 2
            else:
                _, text1, text2 = l.strip().split('\t')
                D.append((text1, text2, 0))
    return D

# 加载数据集
train_data = load_data(
    './datasets/RTE/train.tsv'
)
valid_data = load_data(
    './datasets/RTE/dev.tsv'
)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text1, text2, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()

    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model_RTE.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )
        
def test_predict(in_file, out_file):
    """输出测试结果到文件
    结果文件可以提交到 https://gluebenchmark.com 评测。
    """
    test_data = load_data_test(in_file)
    test_generator = data_generator(test_data, batch_size)

    results = []
    for x_true, _ in tqdm(test_generator, ncols=0):
        y_pred = model.predict(x_true).argmax(axis=1)
        results.extend(y_pred)
        
    with open(out_file,'w',encoding='utf-8') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        csv_writer.writerow(["index","prediction"])
        # 写入tsv文件内容
        for i, pred in enumerate(results):
            csv_writer.writerow([i,labels[pred]])
        # 关闭文件
    f.close()

if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )
    model.load_weights('best_model_RTE.weights')
    #   预测测试集，输出到结果文件
    test_predict(
        in_file = './datasets/RTE/test.tsv',
        out_file = './results/RTE.tsv'
    )
else:

    model.load_weights('best_model_RTE.weights')
