# -*- coding: utf-8 -*-
import re
import numpy as np
import random

f = open('../demo', 'r')

t_source_list = []
t_target_list = []
num_lines = 0
TRAINING_NUM = 10000
VALIDATING_NUM = 10000

# separete the words with the pronunciations, and
# change all the characters in words to lowercase
for line in f.readlines():
    num_lines += 1
    t = line.split()[0].lower()
    if re.match('^[a-z]+[\'\.]?[a-z.]+$', t):
        t_source_list.append(t)
        t_target_list.append(' '.join(line.split()[1:]))
    elif re.match('^([a-z]+[\'\.]?[a-z.]+)\(2\)$', t):
        tt = re.match('^([a-z]+[\'\.]?[a-z.]+)(\(2\))$', t).group(1)
        t_source_list.append(tt)
        # print line.split()
        t_target_list.append(' '.join(line.split()[1:]))


training = open('./training', 'w')
testing = open('./testing', 'w')
validation = open('./validation', 'w')
whole = open('./whole', 'w')

# 经过上面代码的处理以后，t_source_list与t_target_list中分别单词与单词
# 对应的音标序列，你需要将全部数据写入whole中，然后将数据打乱，抽取10000个
# 写入testing，抽取10000个写入validation，剩下的写入training中。
# 单词，音标之间用空格分离，每个单词一行。
# 如： enabled EH0 N EY1 B AH0 L D

# 将单词，音标全部存入文件whole
print(num_lines)
total = []
for i in xrange(0, num_lines):
    temp = t_source_list[i] + ' ' + t_target_list[i]
    total.append(temp)
    whole.write(temp+'\n')
whole.close()

# 随机抽取10000个存入测试文件testing
print(len(total))
result_testing = []
while len(set(result_testing)) <= TRAINING_NUM:
    rnd_line = random.randint(0, num_lines-1)
    result_testing.append(total[rnd_line])

result_set = set(result_testing)
print(len(result_set))
for line in result_set:
    testing.write(line+'\n')
testing.close()

# 除去刚才的10000个测试数据，在余下的数据中随机抽取10000个存入验证文件validation
total_set = set(total)
total_set = total_set - result_set
print(len(total_set))

total_list = list(total_set)
num_lines = len(total_list)
result_validation = []
while len(set(result_validation)) <= VALIDATING_NUM:
    rnd_line = random.randint(0, num_lines-1)
    result_validation.append(total_list[rnd_line])

result_set = set(result_validation)
print(len(result_set))
for line in result_set:
    validation.write(line+'\n')
validation.close()

# 其余的所有数据存入训练集文件中
total_set = total_set - result_set
for line in total_set:
    training.write(line+'\n')
training.close()
