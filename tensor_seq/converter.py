# -*- coding: utf-8 -*-
import os

# 完成以下函数
def extract_words(dict_path, source_path, target_path, file_name):
    """读取数据，并将数据分为单词与读音，存至对应的目录内。
       Args:
           dict_path: 读取数据路径
           source_path: 单词存放路径
           target_path: 发音序列存放路径
           file_name: 文件名前缀

       其中单词文件为每一行为一个单词：
       cheered
       benshoof
       achieve
       
       发音文件为每行一个单词的发音：
       CH IH1 R D
       B EH1 N SH UH0 F
       AH0 CH IY1 V
    """
    file_r = dict_path + file_name
    read_file = open(file_r, 'r')
    file_w_source = source_path + file_name
    file_w_target = target_path + file_name
    source_file = open(file_w_source, 'w')
    target_file = open(file_w_target, 'w')
    for line in read_file.readlines():
        word = line.split()[0]
        pronunciation = ' '.join(line.split()[1:])
        source_file.write(word+'\n')
        target_file.write(pronunciation+'\n')

    read_file.close()
    source_file.close()
    target_file.close()


data_set_path = './dataset/'
if not os.path.exists(data_set_path):
    os.makedirs(data_set_path)

dict_path_pre = '../Split_Dataset/'
source_path_pre = data_set_path + 'source_list_'
target_path_pre = data_set_path + 'target_list_'
extract_words(dict_path_pre, source_path_pre, target_path_pre, 'training')
extract_words(dict_path_pre, source_path_pre, target_path_pre, 'testing')
extract_words(dict_path_pre, source_path_pre, target_path_pre, 'validation')
extract_words(dict_path_pre, source_path_pre, target_path_pre, 'whole')