#! -*- coding:utf-8 -*-
# https://github.com/nishiwen1214/GLUE-bert4keras
# 句子对回归任务，STS-B数据集
# pearson_corr:89.31 spearman_corr:88.90

import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense
from scipy.stats import pearsonr, spearmanr
import numpy as np

set_gelu('tanh')  # 切换gelu版本

maxlen = 128
batch_size = 32
config_path = './uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './uncased_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签y)
    """
    D = []
    i = 1
    with open(filename, encoding='utf-8') as f:
        for l in f:
            if i == 1: # 跳过数据第一行
                i = 2
            else:
                _,_,_,_,_,_,_, text1, text2, label = l.strip().split('\t')
                D.append((text1, text2, float(label)))
    return D

# 加载数据集
train_data = load_data(
    './datasets/STS-B/train.tsv'
)
valid_data = load_data(
    './datasets/STS-B/dev.tsv'
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
    units=1, kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='mse',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    metrics=['mse'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

def evaluate(data):
    y_true_all = np.array([], dtype=int)
    y_pred_all = np.array([], dtype=int)
    for x_true, y_true in data:
        y_pred = model.predict(x_true)
        y_pred_all = np.append(y_pred_all, y_pred)
        y_true_all = np.append(y_true_all, y_true)

    pearson_corr = pearsonr(y_pred_all, y_true_all)[0]
    spearman_corr = spearmanr(y_pred_all, y_true_all)[0]
    return pearson_corr, spearman_corr



class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_pearson_corr = 0.

    def on_epoch_end(self, epoch, logs=None):
        pearson_corr, spearman_corr = evaluate(valid_generator)
        if pearson_corr > self.best_val_pearson_corr:
            self.best_val_pearson_corr = pearson_corr
            model.save_weights('best_model_STS-B.weights')
        print(
            u'pearson_corr: %.5f, spearman_corr: %.5f, best_pearson_corr: %.5f\n' %
            (pearson_corr,spearman_corr, self.best_val_pearson_corr)
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

    model.load_weights('best_model_STS-B.weights')