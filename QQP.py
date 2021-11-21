#! -*- coding:utf-8 -*-
# https://github.com/nishiwen1214/GLUE-bert4keras
# 句子对分类任务，QQP数据集
# val_acc: 88.7071, test_acc: 87.0320

import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense
from sklearn import metrics
import numpy as np

set_gelu('tanh')  # 切换gelu版本

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
                _,_,_, text1, text2, label = l.strip().split('\t')
                D.append((text1, text2, int(label)))
    return D

# 加载数据集
train_data = load_data(
    './datasets/QQP/train.tsv'
)
valid_data = load_data(
    './datasets/QQP/dev.tsv'
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
    with_pool=True,
    return_keras_model=False,
)

output = Dropout(rate=0.1)(bert.model.output)
output = Dense(
    units=2, activation='softmax', kernel_initializer=bert.initializer
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
    y_true_all = np.array([], dtype=int)
    y_pred_all = np.array([], dtype=int)
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        y_pred_all = np.append(y_pred_all, y_pred)
        y_true_all = np.append(y_true_all, y_true)
        total += len(y_true)
        right += (y_true == y_pred).sum()
    f1 = metrics.f1_score(y_true_all,y_pred_all)
    return right / total, f1


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc, f1 = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model_QQP.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, F1: %.5f\n' %
            (val_acc, self.best_val_acc, f1)
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

    model.load_weights('best_model_QQP.weights')