# -*- coding: utf-8 -*-
"""Train or test the model"""

from tensor_seq.model_pron import *

train_source = None
train_target = None
valid_source = None
valid_target = None
test_source = None
test_target = None

if isTrain == 1:
    # 　训练模型，需要训练数据集和测试数据集
    train_remainder = len(data_sets['train'][0]) % batch_size
    valid_remainder = len(data_sets['dev'][0]) % batch_size

    train_source = data_sets['train'][0] + data_sets['train'][0][len(data_sets['train'][0]) - train_remainder - 1:]
    train_target = data_sets['train'][1] + data_sets['train'][1][len(data_sets['train'][0]) - train_remainder - 1:]

    valid_source = data_sets['dev'][0] + data_sets['dev'][0][0:batch_size - valid_remainder]
    valid_target = data_sets['dev'][1] + data_sets['dev'][1][0:batch_size - valid_remainder]

elif isTrain == 2:
    # 验证模型，利用验证数据集
    test_remainder = len(data_sets['test']) % batch_size
    test_source = data_sets['test'][0] + data_sets['test'][0][len(data_sets['test'][0]) - test_remainder - 1:]
    test_target = data_sets['test'][1] + data_sets['test'][1][len(data_sets['test'][0]) - test_remainder - 1:]


# TODO
# 完成 pad_sentence_batch　函数
def pad_sentence_batch(sentence_batch, pad_int):
    """给mini batch增加占位符，使得当前的mini batch中的序列都具有相同的长度
    Args:
        sentence_batch: 要补充占位符的mini-batch
        pad_int: <PAD>对应的整数
    Returns:
        经过补充占位符后的mini-batch
    """
    max_sentence_length = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence_length - len(sentence)) for sentence in sentence_batch]


def get_batches(targets, sources, source_pad_int, target_pad_int):
    """Generator to generating the mini batches for training and testing
    Args:
        targets: targets(true result) used for training the decoder, tensor of shape
          [batch_size, max_target_sequence_length].
        sources: input of the model, tensor of shape [batch_size, max_input_length].
        source_pad_int: an integer representing the symbol of <PAD> for input sequence.
        target_pad_int: an integer representing the symbol of <PAD> for output sequence.
    Yields:
        pad_targets_batch: padded targets mini-batch
        pad_sources_batch: padded inputs mini-batch
        targets_length: tensor of shape (mini_batch_size, ), representing the length for
          each target sequence in the mini-batch.
        source_lengths: tensor of shape (mini_batch_size, ), representing the length for
          each input sequence in the mini-batch.
    """
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths


# create the compute graph
train_graph = tf.Graph()
with train_graph.as_default():
    # define the global step of the graph
    global_step = tf.train.create_global_step(train_graph)
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()

    # define the placeholder and summary of the validation loss and WER
    average_vali_loss = tf.placeholder(dtype=tf.float32)
    WER_over_validation = tf.placeholder(dtype=tf.float32)
    v_c = tf.summary.scalar("validation_cost", average_vali_loss)
    v_wer = tf.summary.scalar("validation_WER", WER_over_validation)

    # get the output of the seq2seq model
    training_decoder_output, predicting_decoder_output, bm_decoder_output = seq2seq_model(input_data,
                                                                                          targets,
                                                                                          target_sequence_length,
                                                                                          max_target_sequence_length,
                                                                                          source_sequence_length,
                                                                                          len(source_letter_to_int),
                                                                                          encoding_embedding_size,
                                                                                          decoding_embedding_size,
                                                                                          rnn_size,
                                                                                          num_layers)
    # get the logits of decoder during training and testing to calculate loss.
    # tf.contrib.seq2seq.dynamic_decode()的输出是training_decoder_output，这个变量有两个属性，
    # 第一个是rnn网络的输出rnn_output，另一个是rnn网络预测的序列sample_id，是一个整数序列
    training_logits = tf.identity(training_decoder_output.rnn_output, 'training_logits')
    predicting_logits = tf.identity(predicting_decoder_output.rnn_output, 'predicting_logits')
    # the result of the prediction
    prediction = tf.identity(predicting_decoder_output.sample_id, 'prediction_result')
    bm_prediction = tf.identity(bm_decoder_output.predicted_ids, 'bm_prediction_result')
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    # the score of the beam search prediction
    bm_score = tf.identity(bm_decoder_output.beam_search_decoder_output.scores, 'bm_prediction_scores')

    with tf.name_scope("optimization"):
        # 计算交叉熵
        train_cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # TODO 定义优化算法
        # 要求optimizer_collection为一个词典，其中
        # key = 0,1,2 分别对应SGD,Adam和RMSProp算法
        optimizer_collection = {0: tf.train.GradientDescentOptimizer,
                                1: tf.train.AdamOptimizer,
                                2: tf.train.RMSPropOptimizer}

        # 使用对应的优化算法
        optimizer = optimizer_collection[optimizer_type]

        # 计算梯度
        gradients = optimizer(learning_rate).compute_gradients(train_cost)
        # 梯度剪裁
        capped_gradients = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gradients if grad is not None]
        # 更新参数
        train_op = optimizer(learning_rate).apply_gradients(capped_gradients, global_step=global_step)
    # define summary to record training cost.
    training_cost_summary = tf.summary.scalar('training_cost', train_cost)

    with tf.name_scope("validation"):
        # get the max length of the predicting result
        val_seq_len = tf.shape(predicting_logits)[1]
        # process the predicting result so that it has the same shape with targets
        predicting_logits = tf.concat([predicting_logits, tf.fill(
            [batch_size, max_target_sequence_length - val_seq_len, tf.shape(predicting_logits)[2]], 0.0)], axis=1)
        # calculate loss
        validation_cost = tf.contrib.seq2seq.sequence_loss(
            predicting_logits,
            targets,
            masks)


# TODO　完成cal_error函数
def cal_error(input_batch, prediction_result):
    """计算一个mini-batch中预测发音的错误的数目
    Args:
        input_batch: 模型输入的mini-batch
        prediction_result: 模型对应的预测结果(整型)
    Returns:
        t_error: mini-batch中预测的错误数
    """
    t_error = 0.0
    for word, pron_result in zip(input_batch, prediction_result):
        flag = 0
        letter_word = []
        for i in word:
            letter_i = source_int_to_letter[i]
            letter_word.append(letter_i)

        word_string = ''
        for i in letter_word:
            if i == '<PAD>':
                break
            word_string += i

        for temp_pron in word_pron[word_string]:
            temp_list_pron = temp_pron.split()
            compare_list_pron = []
            for i in temp_list_pron:
                compare_list_pron.append(target_letter_to_int[i])
            compare_array_pron = np.array(compare_list_pron)

            length = 0
            if len(compare_array_pron) == len(pron_result):
                for j in range(len(pron_result)):
                    length += 1
                    if compare_array_pron[j] != pron_result[j]:
                        break
            if len(compare_array_pron) == length:
                flag = 1
                break
        if flag == 0:
            t_error += 1

    return t_error


# create session to run the TensorFlow operations
with tf.Session(graph=train_graph) as sess:
    # define summary file writer
    t_s = tf.summary.FileWriter('./graph/training', sess.graph)
    v_s = tf.summary.FileWriter('./graph/validation', sess.graph)

    # define saver, keep max_model_number of most recent models
    saver = tf.train.Saver(max_to_keep=max_model_number)

    if isTrain == 1:
        # run initializer
        sess.run(tf.global_variables_initializer())

        # train the model
        for epoch_i in range(1, epochs + 1):
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                    get_batches(train_target, train_source,
                                source_letter_to_int['<PAD>'],
                                target_letter_to_int['<PAD>'])):
                # get global step
                step = tf.train.global_step(sess, global_step)
                t_c, _, loss = sess.run(
                    [training_cost_summary, train_op, train_cost],
                    {input_data: sources_batch,
                     targets: targets_batch,
                     lr: learning_rate,
                     target_sequence_length: targets_lengths,
                     source_sequence_length: sources_lengths})

                if batch_i % display_step == 0:
                    # calculate the word error rate (WER) and validation loss of the model
                    error = 0.0
                    vali_loss = []
                    for _, (
                            valid_targets_batch, valid_sources_batch, valid_targets_lengths,
                            valid_source_lengths) in enumerate(
                        get_batches(valid_target, valid_source,
                                    source_letter_to_int['<PAD>'],
                                    target_letter_to_int['<PAD>'])
                    ):
                        validation_loss, basic_prediction = sess.run(
                            [validation_cost, prediction],
                            {input_data: valid_sources_batch,
                             targets: valid_targets_batch,
                             lr: learning_rate,
                             target_sequence_length: valid_targets_lengths,
                             source_sequence_length: valid_source_lengths})

                        vali_loss.append(validation_loss)

                        # TODO
                        # 计算整个验证数据集中错误输出的个数
                        error += cal_error(valid_sources_batch, basic_prediction)

                    # 用验证数据集对模型进行评估，计算交叉熵，错误率
                    vali_loss = sum(vali_loss) / len(vali_loss)

                    # TODO 计算错误率，表示为WER
                    WER = error/len(valid_target)
                    vali_summary, wer_summary = sess.run([v_c, v_wer], {average_vali_loss: vali_loss,
                                                                        WER_over_validation: WER
                                                                        })

                    # write the cost to summery
                    t_s.add_summary(t_c, global_step=step)
                    v_s.add_summary(vali_summary, global_step=step)
                    v_s.add_summary(wer_summary, global_step=step)

                    print(
                        'Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  '
                        '- Validation loss: {:>6.3f}'
                        ' - WER: {:>6.2%} '.format(epoch_i,
                                                   epochs,
                                                   batch_i,
                                                   len(train_source) // batch_size,
                                                   loss,
                                                   vali_loss,
                                                   WER))
            # save the model every epoch
            saver.save(sess, save_path='./model/model.ckpt', global_step=step)
            print('$$$$$$$$$$$$$$$$$$$$')
            print("Saver the %d time." % epoch_i)
            print('$$$$$$$$$$$$$$$$$$$$')
        # save the model when finished
        saver.save(sess, save_path='./model/model.ckpt', global_step=step)
        print('Model Trained and Saved')

    else:
        # load model from folder
        checkpoint = tf.train.latest_checkpoint('./model')
        saver.restore(sess, checkpoint)
        error = 0.0
        test_loss = []
        for _, (
                test_targets_batch, test_sources_batch, test_targets_lengths,
                test_source_lengths) in enumerate(
            get_batches(test_target, test_source,
                        source_letter_to_int['<PAD>'],
                        target_letter_to_int['<PAD>'])
        ):
            validation_loss, basic_prediction = sess.run(
                [validation_cost, prediction],
                {input_data: test_sources_batch,
                 targets: test_targets_batch,
                 lr: learning_rate,
                 target_sequence_length: test_targets_lengths,
                 source_sequence_length: test_source_lengths})

            test_loss.append(validation_loss)
            # TODO
            # 计算整个测试数据集中错误输出的个数
            error += cal_error(test_sources_batch, basic_prediction)

        test_loss = sum(test_loss) / len(test_loss)
        # TODO
        # 计算WER
        WER = error / len(test_target)
        print('Test loss: {:>6.3f}'
              ' - WER: {:>6.2%} '.format(test_loss, WER))
