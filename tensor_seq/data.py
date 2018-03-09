# -*- coding: utf-8 -*-
"""Map the input and output sequence to int and write to file"""
import pickle

# open the file contains data
with open('./dataset/source_list_whole') as f:
    source_data = f.read().split('\n')
with open('./dataset/target_list_whole') as f:
    target_data = f.read().split('\n')

# 建立词典与发音的预测
word_pron = {}
for word, pron in zip(source_data, target_data):
    if word in word_pron:
        word_pron[word].add(pron)
    else:
        word_pron[word] = {pron}


# map the character and pronunciation to idx
def word2index(data):
    """将单词中的每一个字符映射到一个整数上。
    Args:
        data: 一个包含单词的list
    Returns:
        idx2word: 一个词典，key是整数，value是对应的字符。
        word2idx: 一个词典，key是对应的字符，value是整数。

        返回的两个词典的映射关系应当保持一致。
    """
    # 特殊字符
    special_words = ['<EOS>', '<GO>', '<PAD>', '<UNK>']
    set_words = list(set([c for line in data for c in line] + special_words))

    # 生成返回词典
    idx2word = {key:value for key, value in enumerate(set_words)}
    word2idx = {value:key for key, value in idx2word.items()}

    return idx2word, word2idx


def phon2index(data):
    """Map each symbol in arpabet and special symbol to idx
    Args:
        data: a list contains the pronunciation sequence for each word.
    Returns:
        idx2word: a dict map each idx to a symbol.
        word2idx: a dict map each symbol to an idx.
    """
    special_words = ['<EOS>', '<GO>', '<PAD>', '<UNK>']
    set_words = list(
        set([pho for line in data for pho in line.split()] + special_words))

    idx2word = {idx: word for idx, word in enumerate(set_words)}
    word2idx = {word: idx for idx, word in idx2word.items()}
    return idx2word, word2idx


source_int_to_letter, source_letter_to_int = word2index(source_data)
target_int_to_letter, target_letter_to_int = phon2index(target_data)


def map_int(data_set_name):
    """Map the data set's input and output to integer.

    Args:
        data_set_name: the name of the data set.
    Returns:
        mapping: (input_mapping, output_mapping)
    """
    with open('./dataset/source_list_' + data_set_name) as f:
        source_data = f.read().split('\n')
    with open('./dataset/target_list_' + data_set_name) as f:
        target_data = f.read().split('\n')
    source_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>'])
                   for letter in line] for line in source_data]
    target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>'])
                   for letter in line.split()] + [target_letter_to_int['<EOS>']] for line in target_data]
    return source_int, target_int


data_sets = {'train': map_int('training'),
             'test': map_int('testing'),
             'dev': map_int('validation')}

# save all the objects to a file.
with open('./dataset/data.pickle', 'wb+') as f:
    pickle.dump(
        [source_int_to_letter, source_letter_to_int, target_int_to_letter, target_letter_to_int, data_sets, word_pron],
        f)
