#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np
tf.set_random_seed(1)
np.random.seed(1234)
import sys


FLAGS = tf.app.flags.FLAGS
#general variables
tf.app.flags.DEFINE_string('embedding_type','BERT','can be: glove, word2vec-cbow, word2vec-SG, fasttext, BERT, BERT_Large, ELMo')
tf.app.flags.DEFINE_integer("year",2015, "year data set [2014]")
tf.app.flags.DEFINE_integer('embedding_dim', 768, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 20, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_float('learning_rate', 0.07, 'learning rate')
tf.app.flags.DEFINE_integer('n_class', 4, 'number of distinct class')   #3 for regular, 4 for adversarial
tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 20, 'max number of tokens per sentence')
tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
tf.app.flags.DEFINE_float('random_base', 0.01, 'initial random base')
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_integer('n_iter', 200, 'number of train iter')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 0.5, 'dropout keep prob')
tf.app.flags.DEFINE_string('t1', 'last', 'type of hidden output')
tf.app.flags.DEFINE_string('t2', 'last', 'type of hidden output')
tf.app.flags.DEFINE_integer('n_layer', 3, 'number of stacked rnn')
tf.app.flags.DEFINE_string('is_r', '1', 'prob')
tf.app.flags.DEFINE_integer('max_target_len', 19, 'max target length')
tf.app.flags.DEFINE_integer('gen_input_dims', 100 , 'dimensionality of random input generator')

# traindata, testdata and embeddings, train path aangepast met ELMo
tf.app.flags.DEFINE_string("train_path_ont", "data/programGeneratedData/GloVetraindata"+str(FLAGS.year)+".txt", "train data path for ont")
tf.app.flags.DEFINE_string("test_path_ont",  "data/programGeneratedData/GloVetestdata"+str(FLAGS.year)+".txt", "formatted test data path")
tf.app.flags.DEFINE_string("train_path", "data/programGeneratedData/" + str(FLAGS.embedding_type) +str(FLAGS.embedding_dim)+'traindata'+str(FLAGS.year)+".txt", "train data path")
tf.app.flags.DEFINE_string("test_path", "data/programGeneratedData/" + str(FLAGS.embedding_type) + str(FLAGS.embedding_dim)+'testdata'+str(FLAGS.year)+".txt", "formatted test data path")
tf.app.flags.DEFINE_string("embedding_path", "data/programGeneratedData/" + str(FLAGS.embedding_type) + str(FLAGS.embedding_dim)+'embedding'+str(FLAGS.year)+".txt", "pre-trained glove vectors file path")
tf.app.flags.DEFINE_string("remaining_test_path_ELMo", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'remainingtestdata'+str(FLAGS.year)+"ELMo.txt", "only for printing")
tf.app.flags.DEFINE_string("remaining_test_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'remainingtestdata'+str(FLAGS.year)+".txt", "formatted remaining test data path after ontology")

#svm traindata, svm testdata
tf.app.flags.DEFINE_string("train_svm_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'trainsvmdata'+str(FLAGS.year)+".txt", "train data path")
tf.app.flags.DEFINE_string("test_svm_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'testsvmdata'+str(FLAGS.year)+".txt", "formatted test data path")
tf.app.flags.DEFINE_string("remaining_svm_test_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'remainingsvmtestdata'+str(FLAGS.year)+".txt", "formatted remaining test data path after ontology")

#hyper traindata, hyper testdata
tf.app.flags.DEFINE_string("hyper_train_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hypertraindata'+str(FLAGS.year)+".txt", "hyper train data path")
tf.app.flags.DEFINE_string("hyper_eval_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hyperevaldata'+str(FLAGS.year)+".txt", "hyper eval data path")

tf.app.flags.DEFINE_string("hyper_svm_train_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hypertrainsvmdata'+str(FLAGS.year)+".txt", "hyper train svm data path")
tf.app.flags.DEFINE_string("hyper_svm_eval_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hyperevalsvmdata'+str(FLAGS.year)+".txt", "hyper eval svm data path")

#external data sources
tf.app.flags.DEFINE_string("pretrain_file", "data/externalData/"+str(FLAGS.embedding_type)+"."+str(FLAGS.embedding_dim)+"d.txt", "pre-trained embedding vectors for non BERT and ELMo")

tf.app.flags.DEFINE_string("train_data", "data/externalData/restaurant_train_"+str(FLAGS.year)+".xml",
                    "train data path")
tf.app.flags.DEFINE_string("test_data", "data/externalData/restaurant_test_"+str(FLAGS.year)+".xml",
                    "test data path")

tf.app.flags.DEFINE_string('method', 'AE', 'model type: AE, AT or AEAT')
tf.app.flags.DEFINE_string('prob_file', 'prob1.txt', 'prob')
tf.app.flags.DEFINE_string('saver_file', 'prob1.txt', 'prob')


def print_config():
    #FLAGS._parse_flags()
    FLAGS(sys.argv)
    print('\nParameters:')
    for k, v in sorted(tf.app.flags.FLAGS.flag_values_dict().items()):
        print('{}={}'.format(k, v))


def loss_func(y, prob):
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    pred_loss = tf.reduce_mean(y * tf.log(prob))
    loss = - tf.reduce_mean(y * tf.log(prob))  + sum(reg_loss)
    return loss

def loss_func_adversarial(prob_real, prob_generated, y_real):
    #making y_generated
    zeros = tf.zeros([FLAGS.batch_size, 3], dtype=tf.float32)
    ones = tf.ones([FLAGS.batch_size, 1], dtype=tf.float32)
    y_generated = tf.concat([zeros, ones], 1)  # batch_size x 4 w. each row [0,0,0,1]

    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss =  - tf.reduce_mean(y_real * tf.log(prob_real)) - tf.reduce_mean(y_generated * tf.log(prob_generated)) + sum(reg_loss)
    return loss

def acc_func(y, prob):
    correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    acc_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
    acc_prob = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return acc_num, acc_prob

def acc_func_adversarial(prob_real, prob_generated, y_real):
    #real samples: percentage/number correctly classified real data samples
    correct_pred_real = tf.equal(tf.argmax(prob_real, 1), tf.argmax(y_real, 1))
    acc_num_real = tf.reduce_sum(tf.cast(correct_pred_real, tf.int32))
    acc_prob_real = tf.reduce_mean(tf.cast(correct_pred_real, tf.float32))

    #generated samples: percentage/number corretcly seen as fake
    #making y_generated
    zeros = tf.zeros([FLAGS.batch_size, 3], dtype=tf.float32)
    ones = tf.ones([FLAGS.batch_size, 1], dtype=tf.float32)
    y_generated = tf.concat([zeros, ones], 1)  # batch_size x 4 w. each row [0,0,0,1]

    correct_pred_generated = tf.equal(tf.argmax(prob_generated, 1), tf.argmax(y_generated, 1))
    acc_num_generated = tf.reduce_sum(tf.cast(correct_pred_generated, tf.int32))
    acc_prob_generated = tf.reduce_mean(tf.cast(correct_pred_generated, tf.float32))
    return  acc_num_real, acc_prob_real, acc_num_generated, acc_prob_generated


def train_func(loss, r, global_step, optimizer=None):
    if optimizer:
        return optimizer(learning_rate=r).minimize(loss, global_step=global_step)
    else:
        return tf.train.AdamOptimizer(learning_rate=r).minimize(loss, global_step=global_step)


def summary_func(loss, acc, test_loss, test_acc, _dir, title, sess):
    summary_loss = tf.summary.scalar('loss' + title, loss)
    summary_acc = tf.summary.scalar('acc' + title, acc)
    test_summary_loss = tf.summary.scalar('loss' + title, test_loss)
    test_summary_acc = tf.summary.scalar('acc' + title, test_acc)
    train_summary_op = tf.summary.merge([summary_loss, summary_acc])
    validate_summary_op = tf.summary.merge([summary_loss, summary_acc])
    test_summary_op = tf.summary.merge([test_summary_loss, test_summary_acc])
    train_summary_writer = tf.summary.FileWriter(_dir + '/train', sess.graph)
    test_summary_writer = tf.summary.FileWriter(_dir + '/test')
    validate_summary_writer = tf.summary.FileWriter(_dir + '/validate')
    return train_summary_op, test_summary_op, validate_summary_op, \
        train_summary_writer, test_summary_writer, validate_summary_writer

def summary_func_adversarial(loss, acc_real, acc_generated, test_loss, test_acc, _dir, title, sess):
    summary_loss = tf.summary.scalar('loss' + title, loss)
    summary_acc_real = tf.summary.scalar('acc real' + title, acc_real)
    summary_acc_generated = tf.summary.scalar('acc generated' + title, acc_generated)
    test_summary_loss = tf.summary.scalar('loss' + title, test_loss)
    test_summary_acc = tf.summary.scalar('acc' + title, test_acc)

    train_summary_op = tf.summary.merge([summary_loss, summary_acc_real, summary_acc_generated])
    validate_summary_op = tf.summary.merge([test_summary_loss, test_summary_acc])
    test_summary_op = tf.summary.merge([test_summary_loss, test_summary_acc])
    train_summary_writer = tf.summary.FileWriter(_dir + '/train', sess.graph)
    test_summary_writer = tf.summary.FileWriter(_dir + '/test')
    validate_summary_writer = tf.summary.FileWriter(_dir + '/validate')
    return train_summary_op, test_summary_op, validate_summary_op, \
           train_summary_writer, test_summary_writer, validate_summary_writer,


def saver_func(_dir):
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1000)
    import os
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return saver