# -*- coding: utf-8 -*-
"""模型相关的参数，超参数"""
# 学习率
learning_rate = 0.01
# 模型使用的优化算法， 0 对应 SGD, 1 对应 Adam, 2 对应 RMSProp
optimizer_type = 1
# mini-batch的大小
batch_size = 512
# RNN结构 0 对应 LSTM, 1 对应 GRU
Cell_type = 0
# 激活函数的种类， 0 对应 tanh, 1 对应 relu, 2 对应 sigmoid
activation_type = 0
# 每层rnn中神经元的个数
rnn_size = 128
# 层数
num_layers = 2
# embedding的大小
encoding_embedding_size = 64
decoding_embedding_size = 128
# Decoder使用的种类，0　使用basic decoder, 1使用beam search
Decoder_type = 0
#　选择beam search decoder　时的　beam width，影响最终结果的个数 
beam_width = 3
# 最大模型训练次数
epochs = 300
# 1是训练模型，2是测试模型
isTrain = 1
# 每隔多少mini-batch输出一次
display_step = 50
# 保存最近几个模型
max_model_number = 3
