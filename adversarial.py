import os, sys
sys.path.append(os.getcwd())

from sklearn.metrics import precision_score, recall_score, f1_score
from nn_layer import softmax_layer, bi_dynamic_rnn, reduce_mean_with_len
from att_layer import bilinear_attention_layer, dot_produce_attention_layer
from config import *
from utils import load_w2v, batch_index, load_inputs_twitter
from att_layer import mlp_layer_woalpha
from datetime import datetime

import tensorflow as tf
import numpy as np
tf.set_random_seed(1)
np.random.seed(1234)

#generator
def generator(l2):
    random_input = tf.random_uniform(shape=[FLAGS.batch_size, FLAGS.gen_input_dims])
    # dimensionality assigns
    n_input = FLAGS.gen_input_dims
    n_hidden_1 = 2 * FLAGS.n_hidden
    n_hidden_2 = 6 * FLAGS.n_hidden
    n_outputs = 8 * FLAGS.n_hidden

    with tf.variable_scope('var_G') as scope:
        # defining weights and biases
        gen_h1 = tf.get_variable(name='gen_h1', shape=[n_input, n_hidden_1],
                                 initializer=tf.random_uniform_initializer(minval=-FLAGS.random_base,
                                                                           maxval=FLAGS.random_base),
                                 regularizer=tf.contrib.layers.l2_regularizer(l2),
                                 dtype=tf.float32)

        gen_h2 = tf.get_variable(name='gen_h2', shape=[n_hidden_1, n_hidden_2],
                                 initializer=tf.random_uniform_initializer(minval=-FLAGS.random_base,
                                                                           maxval=FLAGS.random_base),
                                 regularizer=tf.contrib.layers.l2_regularizer(l2),
                                 dtype=tf.float32)

        gen_h_out = tf.get_variable(name='gen_h_out', shape=[n_hidden_2, n_outputs],
                                    initializer=tf.random_uniform_initializer(minval=-FLAGS.random_base,
                                                                              maxval=FLAGS.random_base),
                                    regularizer=tf.contrib.layers.l2_regularizer(l2),
                                    dtype=tf.float32)

        gen_b1 = tf.get_variable(name='gen_b1', shape=[n_hidden_1],
                                 initializer=tf.zeros_initializer(),
                                 regularizer=tf.contrib.layers.l2_regularizer(l2),
                                 dtype=tf.float32)

        gen_b2 = tf.get_variable(name='gen_b2', shape=[n_hidden_2],
                                 initializer=tf.zeros_initializer(),
                                 regularizer=tf.contrib.layers.l2_regularizer(l2),
                                 dtype=tf.float32)

        gen_b_out = tf.get_variable(name='gen_b_out', shape=[n_outputs],
                                    initializer=tf.zeros_initializer(),
                                    regularizer=tf.contrib.layers.l2_regularizer(l2),
                                    dtype=tf.float32)

    print('I am the Generator')
    # 2 layer multilayer perceptron
    layer_1 = tf.add(tf.matmul(random_input, gen_h1), gen_b1)
    layer_2 = tf.add(tf.matmul(layer_1, gen_h2), gen_b2)
    out_layer = tf.add(tf.matmul(layer_2, gen_h_out), gen_b_out)
    # splitting output into 4 attention vectors.
    gen_l, gen_r, gen_t_l, gen_t_r = tf.split(out_layer, num_or_size_splits=4, axis=1)

    return gen_l, gen_r, gen_t_l, gen_t_r

#lcr-rot minus the final MLP layer
def lcr_rot (input_fw, input_bw, sen_len_fw, sen_len_bw, target, sen_len_tr, keep_prob1, keep_prob2, l2, _id='all'):

    print('I am lcr_rot_alt_Adversarial.')
    cell = tf.contrib.rnn.LSTMCell
    # left hidden
    input_fw = tf.nn.dropout(input_fw, keep_prob=keep_prob1,seed=1)
    hiddens_l = bi_dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'var_D' + '_lh', 'all')
    pool_l = reduce_mean_with_len(hiddens_l, sen_len_fw)

    # right hidden
    input_bw = tf.nn.dropout(input_bw, keep_prob=keep_prob1,seed=2)
    hiddens_r = bi_dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'var_D' + '_rh', 'all')
    pool_r = reduce_mean_with_len(hiddens_r, sen_len_bw)

    # target hidden
    target = tf.nn.dropout(target, keep_prob=keep_prob1,seed=3)
    hiddens_t = bi_dynamic_rnn(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_sentence_len, 'var_D' + '_th', 'all')
    pool_t = reduce_mean_with_len(hiddens_t, sen_len_tr)

    # attention left
    att_l = bilinear_attention_layer(hiddens_l, pool_t, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'var_D' + '_al_init', 'tl')
    outputs_t_l_init = tf.matmul(att_l, hiddens_l)
    outputs_t_l = tf.squeeze(outputs_t_l_init)
    # attention right
    att_r = bilinear_attention_layer(hiddens_r, pool_t, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'var_D' + '_ar_init', 'tr')
    outputs_t_r_init = tf.matmul(att_r, hiddens_r)
    outputs_t_r = tf.squeeze(outputs_t_r_init)

    # attention target left
    att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                        'var_D' + '_atl', 'l')
    outputs_l_init = tf.matmul(att_t_l, hiddens_t)
    outputs_l = tf.squeeze(outputs_l_init)
    # attention target right
    att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                        'var_D' + '_atr', 'r')
    outputs_r_init = tf.matmul(att_t_r, hiddens_t)
    outputs_r = tf.squeeze(outputs_r_init)

    outputs_init_context = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
    outputs_init_target = tf.concat([outputs_l_init, outputs_r_init], 1)
    att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2,
                                                      FLAGS.random_base, 'var_D' + '_aoc_init', 'fin1')
    att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2,
                                                     FLAGS.random_base, 'var_D' + '_aot_init', 'fin2')
    outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 0], 2), outputs_l_init))
    outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 1], 2), outputs_r_init))
    outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 0], 2), outputs_t_l_init))
    outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 1], 2), outputs_t_r_init))

    for i in range(2):
        # attention target
        att_l = bilinear_attention_layer(hiddens_l, outputs_l, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                         'var_D' + '_al_' + str(i) , 'tl' + str(i))
        outputs_t_l_init = tf.matmul(att_l, hiddens_l)
        outputs_t_l = tf.squeeze(outputs_t_l_init)

        att_r = bilinear_attention_layer(hiddens_r, outputs_r, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                         'var_D'+'_ar_' + str(i), 'tr' + str(i))
        outputs_t_r_init = tf.matmul(att_r, hiddens_r)
        outputs_t_r = tf.squeeze(outputs_t_r_init)

        # attention left
        att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, l2,
                                           FLAGS.random_base,'var_D'+'_atl_' + str(i), 'l' + str(i))
        outputs_l_init = tf.matmul(att_t_l, hiddens_t)
        outputs_l = tf.squeeze(outputs_l_init)

        # attention right
        att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, l2,
                                           FLAGS.random_base,'var_D'+'_atr_' + str(i), 'r' + str(i))
        outputs_r_init = tf.matmul(att_t_r, hiddens_t)
        outputs_r = tf.squeeze(outputs_r_init)

        outputs_init_context = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
        outputs_init_target = tf.concat([outputs_l_init, outputs_r_init], 1)
        att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2,
                                                          FLAGS.random_base, 'var_D' + '_aoc_final_' + str(i) , 'fin1' + str(i))
        att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2,
                                                         FLAGS.random_base,'var_D'+ '_aot_final_' + str(i), 'fin2' + str(i))
        outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 0], 2), outputs_l_init))
        outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 1], 2), outputs_r_init))
        outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 0], 2), outputs_t_l_init))
        outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 1], 2), outputs_t_r_init))

        out_layer = tf.concat([outputs_l, outputs_r, outputs_t_l, outputs_t_r], 1)
    return outputs_l, outputs_r, outputs_t_l, outputs_t_r, l2, att_l, att_r, att_t_l, att_t_r

#final MLP layer / serving as discriminator
def discriminator(l, r, t_l, t_r, keep_prob2, l2):

    with tf.variable_scope('var_D') as scope:
        w_discriminator = tf.get_variable(
            name='discriminator_w',
            shape=[8*FLAGS.n_hidden,FLAGS.n_class],
            initializer=tf.random_uniform_initializer(-FLAGS.random_base, FLAGS.random_base),
            regularizer=tf.contrib.layers.l2_regularizer(l2)
        )
        b_discriminator = tf.get_variable(
            name='discriminator_b',
            shape=[FLAGS.n_class],  #FLAGS.batch_size,
            initializer=tf.zeros_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(l2)
        )

    print('I am the Discriminator')
    # set-up as the 3-classifier MLP of the original LCR-rot
    outputs_fin = tf.concat([l, r, t_l, t_r], 1)
    outputs = tf.nn.dropout(outputs_fin, keep_prob=keep_prob2,seed=1234)
    predict = tf.matmul(outputs,w_discriminator) +  b_discriminator
    prob = tf.nn.softmax(predict)

    return prob


#main                                                                   #original = 0.09
def main(train_path, test_path, accuracyOnt, test_size, remaining_size, learning_rate_dis, learning_rate_gen,
         keep_prob, momentum_dis, momentum_gen , l2, k, WriteFile):
    print_config()
    with tf.device('/gpu:1'):
        word_id_mapping, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
        word_embedding = tf.constant(w2v, name='word_embedding')

        keep_prob1 = tf.placeholder(tf.float32)
        keep_prob2 = tf.placeholder(tf.float32)

        with tf.name_scope('inputs'):
            x_real = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            y_real = tf.placeholder(tf.float32, [None, FLAGS.n_class])
            sen_len = tf.placeholder(tf.int32, None)

            x_bw = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            sen_len_bw = tf.placeholder(tf.int32, [None])

            target_words = tf.placeholder(tf.int32, [None, FLAGS.max_target_len])
            tar_len = tf.placeholder(tf.int32, [None])


        inputs_fw = tf.nn.embedding_lookup(word_embedding, x_real)
        inputs_bw = tf.nn.embedding_lookup(word_embedding, x_bw)
        target = tf.nn.embedding_lookup(word_embedding, target_words)

        l, r, t_l, t_r, l2, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r = lcr_rot(inputs_fw, inputs_bw, sen_len,
                                                                               sen_len_bw, target, tar_len, keep_prob1,
                                                                               keep_prob2, l2, 'all')
        gen_l, gen_r, gen_t_l, gen_t_r = generator(l2)

        with tf.variable_scope("var_D", reuse=tf.AUTO_REUSE) as scope:      #re-using the discriminator parameters since it is called twice per iter
            #Calculating prob for real data
            prob_real = discriminator(l, r, t_l, t_r, keep_prob2, l2)

            #Calculating prob for generated data
            prob_generated = discriminator(gen_l, gen_r, gen_t_l, gen_t_r, keep_prob2, l2)

        loss = loss_func_adversarial(prob_real, prob_generated, y_real)
        acc_num_real, acc_prob_real, acc_num_gen, acc_prob_gen = acc_func_adversarial(prob_real, prob_generated, y_real)
        global_step = tf.Variable(0, name='tr_global_step', trainable=False)

        #set variable lists
        var_list_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='var_D')
        var_list_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='var_G')

        #As we solve a min max problem, we optimize twice with respect to different variable sets , var_list = var_D , var_list = var_G
        opti_min = tf.train.MomentumOptimizer(learning_rate=learning_rate_dis, momentum=momentum_dis).minimize(loss, var_list = var_list_D,
                                                                                                        global_step=global_step)
        opti_max = tf.train.MomentumOptimizer(learning_rate=learning_rate_gen, momentum=momentum_gen).minimize(-loss, var_list = var_list_G)

        true_y = tf.argmax(y_real, 1)
        pred_y = tf.argmax(prob_real, 1)

        title = '-d1-{}d2-{}b-{}rd-{}rg-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
            FLAGS.keep_prob1,
            FLAGS.keep_prob2,
            FLAGS.batch_size,
            learning_rate_dis,
            learning_rate_gen,
            FLAGS.l2_reg,
            FLAGS.max_sentence_len,
            FLAGS.embedding_dim,
            FLAGS.n_hidden,
            FLAGS.n_class
        )


    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        import time
        timestamp = str(int(time.time()))
        _dir = 'summary/' + str(timestamp) + '_' + title
        test_loss = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)
        train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
        validate_summary_writer = summary_func_adversarial(loss, acc_prob_real, acc_prob_gen, test_loss, test_acc, _dir, title, sess)
        save_dir = 'temp_model/' + str(timestamp) + '_' + title + '/'
        # saver = saver_func(save_dir)


        # saver.restore(sess, '/-')

        if FLAGS.is_r == '1':
             is_r = True
        else:
             is_r = False

        tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len, _, _, _ = load_inputs_twitter(
            train_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len
        )
        te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, te_tar_len, _, _, _ = load_inputs_twitter(
            test_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len
        )

        def get_batch_data(x_f, sen_len_f, x_b, sen_len_b, yi, target, tl, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yi), batch_size, 1, is_shuffle):
                feed_dict = {
                    x_real: x_f[index],
                    x_bw: x_b[index],
                    y_real: yi[index],
                    sen_len: sen_len_f[index],
                    sen_len_bw: sen_len_b[index],
                    target_words: target[index],
                    tar_len: tl[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        max_acc = 0.
        max_fw, max_bw = None, None
        max_tl, max_tr = None, None
        max_ty, max_py = None, None
        max_prob = None
        step = None

        Results_File = np.zeros((5,1)) #6 = number of rows / values to store:['Iteration','loss','trainacc_real','test_acc','avg prob assigned to correct generated']
        for i in range(1,FLAGS.n_iter+1):
            avg_p_real = None
            avg_p_gen = None

            #update D more often than G
            if k >= 1:
                if i%k == 0:
                    print('In iter '+str(i)+' we update both G and D.')
                    trainacc_real, trainacc_gen, traincnt = 0., 0., 0
                    for train, numtrain in get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y,
                                                          tr_target_word,
                                                          tr_tar_len,
                                                          FLAGS.batch_size, keep_prob, keep_prob):
                        # _, step = sess.run([optimizer, global_step], feed_dict=train)

                        _, _, step, summary, _trainacc_real, _trainacc_gen = sess.run(
                            [opti_max, opti_min, global_step, train_summary_op, acc_num_real, acc_num_gen],
                            feed_dict=train)
                        train_summary_writer.add_summary(summary, step)
                        # embed_update = tf.assign(word_embedding, tf.concat(0, [tf.zeros([1, FLAGS.embedding_dim]), word_embedding[1:]]))
                        # sess.run(embed_update)
                        trainacc_real += _trainacc_real  # saver.save(sess, save_dir, global_step=step)
                        trainacc_gen += _trainacc_gen
                        traincnt += numtrain
                else:
                    print('In iter '+str(i)+' we update only D.')
                    trainacc_real, trainacc_gen, traincnt = 0., 0., 0
                    for train, numtrain in get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word,
                                                          tr_tar_len,
                                                          FLAGS.batch_size, keep_prob, keep_prob):
                        # _, step = sess.run([optimizer, global_step], feed_dict=train)

                        _,step, summary, _trainacc_real, _trainacc_gen = sess.run([opti_min,global_step, train_summary_op, acc_num_real, acc_num_gen],
                                                               feed_dict=train)
                        train_summary_writer.add_summary(summary, step)
                        # embed_update = tf.assign(word_embedding, tf.concat(0, [tf.zeros([1, FLAGS.embedding_dim]), word_embedding[1:]]))
                        # sess.run(embed_update)
                        trainacc_real += _trainacc_real  # saver.save(sess, save_dir, global_step=step)
                        trainacc_gen += _trainacc_gen
                        traincnt += numtrain

            #Update G more often than D
            else:
                k_inv = 1/k
                if i%k_inv == 0:
                    print('In iter '+str(i)+' we update both G and D.')
                    trainacc_real, trainacc_gen, traincnt = 0., 0., 0
                    for train, numtrain in get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y,
                                                          tr_target_word,
                                                          tr_tar_len,
                                                          FLAGS.batch_size, keep_prob, keep_prob):
                        # _, step = sess.run([optimizer, global_step], feed_dict=train)

                        _, _, step, summary, _trainacc_real, _trainacc_gen = sess.run(
                            [opti_max, opti_min, global_step, train_summary_op, acc_num_real, acc_num_gen],
                            feed_dict=train)
                        train_summary_writer.add_summary(summary, step)
                        # embed_update = tf.assign(word_embedding, tf.concat(0, [tf.zeros([1, FLAGS.embedding_dim]), word_embedding[1:]]))
                        # sess.run(embed_update)
                        trainacc_real += _trainacc_real  # saver.save(sess, save_dir, global_step=step)
                        trainacc_gen += _trainacc_gen
                        traincnt += numtrain
                else:
                    print('In iter '+str(i)+' we update only G.')
                    trainacc_real, trainacc_gen, traincnt = 0., 0., 0
                    for train, numtrain in get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word,
                                                          tr_tar_len,
                                                          FLAGS.batch_size, keep_prob, keep_prob):
                        # _, step = sess.run([optimizer, global_step], feed_dict=train)

                        _,step, summary, _trainacc_real, _trainacc_gen = sess.run([opti_max,global_step, train_summary_op, acc_num_real, acc_num_gen],
                                                               feed_dict=train)
                        train_summary_writer.add_summary(summary, step)
                        # embed_update = tf.assign(word_embedding, tf.concat(0, [tf.zeros([1, FLAGS.embedding_dim]), word_embedding[1:]]))
                        # sess.run(embed_update)
                        trainacc_real += _trainacc_real  # saver.save(sess, save_dir, global_step=step)
                        trainacc_gen += _trainacc_gen
                        traincnt += numtrain

            #Testing occurs in every iteration, regardless of what networks have been updated.
            acc, cost, cnt = 0., 0., 0
            fw, bw, tl, tr, ty, py = [], [], [], [], [], []
            p = []
            for test, num in get_batch_data(te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y,
                                            te_target_word, te_tar_len, 2000, 1.0, 1.0, False):
                if FLAGS.method == 'TD-ATT' or FLAGS.method == 'IAN':
                    _loss, _acc, _fw, _bw, _tl, _tr, _ty, _py, _p = sess.run(
                        [loss, acc_num, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r, true_y, pred_y, prob_real], feed_dict=test)
                    fw += list(_fw)
                    bw += list(_bw)
                    tl += list(_tl)
                    tr += list(_tr)
                else:
                    _loss, _acc, _ty, _py, _p, _fw, _bw, _tl, _tr,_p_g,_y_real,_prob_real = sess.run(
                        [loss, acc_num_real, true_y, pred_y, prob_real, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r,prob_generated,y_real,prob_real],
                        feed_dict=test)

                ty = np.asarray(_ty)
                py = np.asarray(_py)
                p = np.asarray(_p)
                fw = np.asarray(_fw)
                bw = np.asarray(_bw)
                tl = np.asarray(_tl)
                tr = np.asarray(_tr)
                yr = np.asarray(y_real)
                acc += _acc
                cost += _loss * num
                cnt += num
                p_g = np.asarray(_p_g)


            print('all samples={}, correct prediction={}'.format(cnt, acc))
            trainacc_real = trainacc_real / traincnt
            trainacc_gen = trainacc_gen / traincnt
            acc = acc / cnt
            totalacc = ((acc * remaining_size) + (accuracyOnt * (test_size - remaining_size))) / test_size
            cost = cost / cnt
            print('Iter {}: mini-batch loss={:.6f}, train acc real ={:.6f}, test acc={:.6f}, combined acc={:.6f}'.format(i,
                                                                                                                   cost,
                                                                                                                   trainacc_real,
                                                                                                                   acc,
                                                                                                                   totalacc))
            summary = sess.run(test_summary_op, feed_dict={test_loss: cost, test_acc: acc})
            test_summary_writer.add_summary(summary, step)
            if acc > max_acc:
                max_acc = acc
                max_fw = fw
                max_bw = bw
                max_tl = tl
                max_tr = tr
                max_ty = ty
                max_py = py
                max_prob = p

            #Writing File
            if WriteFile:
                avg_p_real = np.mean(np.multiply(_prob_real,_y_real))       #average probability assigned to the correct class for real data
                avg_p_gen = np.mean(p_g,axis=0)[3]                          #average probability assigned to the correct class for generated data
                Added = [[i],[cost],[trainacc_real],[acc],[avg_p_gen]]
                Results_File = np.concatenate((Results_File,Added),1)

            if np.isnan(_loss):
                print('Ohw shit we obtained an NaN bro!!')
                max_acc = max_acc*((i/200)**2) #Uncomment this line for hyperpar optim, to penalize
                break

        P = precision_score(max_ty, max_py, average=None)
        R = recall_score(max_ty, max_py, average=None)
        F1 = f1_score(max_ty, max_py, average=None)
        print('P:', P, 'avg=', sum(P) / FLAGS.n_class)
        print('R:', R, 'avg=', sum(R) / FLAGS.n_class)
        print('F1:', F1, 'avg=', sum(F1) / FLAGS.n_class)

        fp = open(FLAGS.prob_file, 'w')
        for item in max_prob:
            fp.write(' '.join([str(it) for it in item]) + '\n')
        fp = open(FLAGS.prob_file + '_fw', 'w')
        for y1, y2, ws in zip(max_ty, max_py, max_fw):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        fp = open(FLAGS.prob_file + '_bw', 'w')
        for y1, y2, ws in zip(max_ty, max_py, max_bw):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        fp = open(FLAGS.prob_file + '_tl', 'w')
        for y1, y2, ws in zip(max_ty, max_py, max_tl):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        fp = open(FLAGS.prob_file + '_tr', 'w')
        for y1, y2, ws in zip(max_ty, max_py, max_tr):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')

        print('Optimization Finished! Max acc={}'.format(max_acc))

        print('Learning_rate_dis={},Learning_rate_gen={}, momentum_dis={},momentum_gen={}, iter_num={}, batch_size={}, hidden_num={}, l2={},k={}'.format(
            learning_rate_dis,
            learning_rate_gen,
            momentum_dis,
            momentum_gen,
            FLAGS.n_iter,
            FLAGS.batch_size,
            FLAGS.n_hidden,
            FLAGS.l2_reg,
            k
        ))

        if WriteFile:
            #Saving training information as csv file
            dateTimeObj = datetime.now()
            save_dir = '/Results_Run_Adversarial/Run_' + str(
                dateTimeObj) + '_lr'+str(learning_rate_dis)+ '_lrg'+str(learning_rate_gen)+'_kp'+ str(keep_prob)+ '_mom_d'+str(momentum_dis) +'_mom_g'+str(momentum_gen) + '_k'+str(k)+'.csv'
            np.savetxt(save_dir, Results_File, delimiter=",")


        return max_acc, np.where(np.subtract(max_py, max_ty) == 0, 0,
                                 1), max_fw.tolist(), max_bw.tolist(), max_tl.tolist(), max_tr.tolist()

    if __name__ == '__main__':
        tf.app.run()