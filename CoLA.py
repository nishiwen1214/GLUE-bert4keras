#! -*- coding:utf-8 -*-
# https://github.com/nishiwen1214/GLUE-bert4keras
# 数据集：CoLA 
# MCC: 60.066
# 适用于Keras 2.3.1

import json
import numpy as np
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import os
# 选择使用第几张GPU卡，'0'为第一张
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

num_classes = 2
maxlen = 128
batch_size = 32

# BERT base
config_path = './uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './uncased_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            _,label,_,text = l.strip().split('\t')
            D.append((text,int(label)))
    return D

# 加载数据集
train_data = load_data(
    './datasets/CoLA/train.tsv'
)
valid_data = load_data(
    './datasets/CoLA/dev.tsv'
)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

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
    optimizer=Adam(2e-5),
    metrics=['accuracy'],
)

def evaluate(data):
    total, right = 0., 0.
    y_true_all = np.array([], dtype=int)
    y_pred_all = np.array([], dtype=int)
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        y_pred_all = np.append(y_pred_all, y_pred)
        y_true_all = np.append(y_true_all, y_true)
        total += len(y_true)
        right += (y_true == y_pred).sum()
 
    mcc = matthews_correlation(y_true_all, y_pred_all)
    return right / total, mcc

def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = K.eval((tp * tn - fp * fn))
    denominator = K.eval(((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))** 0.5

    return numerator / (denominator + K.epsilon())

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_mcc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc, mcc = evaluate(valid_generator)
        if mcc > self.best_val_mcc:
            self.best_val_mcc = mcc
            model.save_weights('best_model_CoLA.weights')
        print(
            u'val_acc: %.5f, Matthews corr: %.5f, best_val_mcc: %.5f\n' %
            (val_acc, mcc, self.best_val_mcc)
        )


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model_CoLA.weights')
