import sys
import time
from Utils import *

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


def attention_reader(rnn_type,hidden_size,dropout_rate_embd,dropout_rate_d,dropout_rate_q,learning_rate,batch_size,optimizer_name,num_epoches,eval_iter,
                     vocab_size, embd_size, glove_embd_w,num_labels, model_path,
                     all_train, all_dev):
    print('-'* 50)
    print('Creating TF computation graph...')

    if rnn_type == 'lstm':
        print('Using LSTM Cells')
    elif rnn_type == 'gru':
        print('Using GRU Cells')

    # tf.reset_default_graph()
    d_input = tf.placeholder(dtype=tf.int32, shape=(None, None), name="d_input")
    q_input = tf.placeholder(dtype=tf.int32, shape=(None, None), name="q_input") # [batch_size, max_seq_length_for_batch]
    l_mask = tf.placeholder(dtype=tf.float32, shape=(None, None), name="l_mask") # [batch_size, entity num]
    answer = tf.placeholder(dtype=tf.int32, shape=None, name="label") # batch size vector
    y_1hot= tf.placeholder(dtype=tf.float32, shape=(None, None), name="label_1hot") # onehot encoding of y [batch_size, entitydict]
    training = tf.placeholder(dtype=tf.bool)

    word_embeddings = tf.get_variable("glove", shape=(vocab_size, embd_size), initializer=tf.constant_initializer(glove_embd_w))

    W_bilinear = tf.Variable(tf.random_uniform((2*hidden_size, 2*hidden_size), minval=-0.01, maxval=0.01))

    with tf.variable_scope('d_encoder'): # Encoding Step for Passage (d_ for document)
        d_embed = tf.nn.embedding_lookup(word_embeddings, d_input) # Apply embeddings: [batch, max passage length in batch, GloVe Dim]
        d_embed_dropout = tf.layers.dropout(d_embed, rate=dropout_rate_embd, training=training) # Apply Dropout to embedding layer
        if rnn_type == 'lstm':
            d_cell_fw = rnn.LSTMCell(hidden_size)
            d_cell_bw = rnn.LSTMCell(hidden_size)
            if training == True:
                d_cell_fw = tf.nn.rnn_cell.DropoutWrapper(d_cell_fw, input_keep_prob=(1-dropout_rate_d))
                d_cell_bw = tf.nn.rnn_cell.DropoutWrapper(d_cell_bw, input_keep_prob=(1-dropout_rate_d))
        elif rnn_type == 'gru':
            d_cell_fw = rnn.GRUCell(hidden_size) 
            d_cell_bw = rnn.GRUCell(hidden_size)
            if training == True:
                d_cell_fw = tf.nn.rnn_cell.DropoutWrapper(d_cell_fw, input_keep_prob=(1-dropout_rate_d))
                d_cell_bw = tf.nn.rnn_cell.DropoutWrapper(d_cell_bw, input_keep_prob=(1-dropout_rate_d))

        d_outputs, _ = tf.nn.bidirectional_dynamic_rnn(d_cell_fw, d_cell_bw, d_embed_dropout, dtype=tf.float32)
        d_output = tf.concat(d_outputs, axis=-1) # [batch, len, h], len is the max passage length, and h is the hidden size

    with tf.variable_scope('q_encoder'): # Encoding Step for Question
        q_embed = tf.nn.embedding_lookup(word_embeddings, q_input)
        q_embed_dropout = tf.layers.dropout(q_embed, rate=dropout_rate_embd, training=training)
        if rnn_type == 'lstm':
            q_cell_fw = rnn.LSTMCell(hidden_size)
            q_cell_bw = rnn.LSTMCell(hidden_size)
            if training == True:
                q_cell_fw = tf.nn.rnn_cell.DropoutWrapper(q_cell_fw, input_keep_prob=(1-dropout_rate_q))
                q_cell_bw = tf.nn.rnn_cell.DropoutWrapper(q_cell_bw, input_keep_prob=(1-dropout_rate_q))
        elif rnn_type == 'gru':
            q_cell_fw = rnn.GRUCell(hidden_size)
            q_cell_bw = rnn.GRUCell(hidden_size)
            if training == True:
                q_cell_fw = tf.nn.rnn_cell.DropoutWrapper(q_cell_fw, input_keep_prob=(1-dropout_rate_q))
                q_cell_bw = tf.nn.rnn_cell.DropoutWrapper(q_cell_bw, input_keep_prob=(1-dropout_rate_q))
        q_outputs, q_laststates = tf.nn.bidirectional_dynamic_rnn(q_cell_fw, q_cell_bw, q_embed_dropout, dtype=tf.float32)
        if rnn_type == 'lstm':
            q_output = tf.concat([q_laststates[0][-1], q_laststates[1][-1]], axis=-1) #(batch, h) different with d_encoder[batch,len,out]
        elif rnn_type == 'gru':
            q_output= tf.concat(q_laststates, axis=-1) # (batch, h)

    with tf.variable_scope('bilinear'): # Bilinear Layer (Attention Step)
        # M computes the similarity between each passage word and the entire question encoding
        M = d_output * tf.expand_dims(tf.matmul(q_output, W_bilinear), axis=1) # [batch, h][?,] -> [batch, 1, h]
        # alpha represents the normalized weights representing how relevant the passage word is to the question
        alpha = tf.nn.softmax(tf.reduce_sum(M, axis=2)) # [batch, len]
        # this output contains the weighted combination of all contextual embeddings
        bilinear_output = tf.reduce_sum(d_output * tf.expand_dims(alpha, axis=2), axis=1) # [batch, h]

    with tf.variable_scope('dense'): # Prediction Step
        # the final output has dimension [batch, entity#], giving the probabilities of an entity being the answer for examples
        final_prob = tf.layers.dense(bilinear_output, units=num_labels, activation=tf.nn.softmax, 
                                     kernel_initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01)) # [batch, entity#]

    pred = final_prob * l_mask # ignore entities that don't appear in the passage
    train_pred = pred / tf.expand_dims(tf.reduce_sum(pred, axis=1), axis=1) # redistribute probabilities ignoring certain labels
    train_pred = tf.clip_by_value(train_pred, 1e-7, 1.0 - 1e-7)

    test_pred = tf.cast(tf.argmax(pred, axis=-1), tf.int32)
    acc = tf.reduce_sum(tf.cast(tf.equal(test_pred, answer), tf.int32)) # this for validation acc
    acc1 = tf.reduce_sum(tf.cast(tf.equal(test_pred, answer), tf.int32)) # this for training acc

    loss_op = tf.reduce_sum(-tf.reduce_sum(y_1hot * tf.log(train_pred), reduction_indices=[1]))
    
    if optimizer_name == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif optimizer_name == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        
    gradients = optimizer.compute_gradients(loss_op)
    capped_gradients = [(tf.clip_by_norm(grad,clip_norm=10.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
    print('Build the Model Done!')
    print('-' * 50)

    print('Initial Test...')
    dev_acc = 0. # TODO: first dev accuracy displays here
    print('Initial Dev Accuracy: %.2f %%' % dev_acc)
    best_acc = dev_acc

    saver = tf.train.Saver()

    #print('-'* 50)
    #print('Testing...')
    #if test_only:
    #    if args.test_file == None:
    #        return ValueError("No test file specified")
    #    test_examples = load_data(test_file)
    #    test_x1, test_x2, test_l, test_y = utils.vectorize(test_examples, word_dict, entity_dict)
    #    all_test = gen_examples(test_x1, test_x2, test_l, test_y, batch_size)
    #    with tf.Session() as sess:
    #        # saver = tf.train.import_meta_graph(model_path + '.meta')
    #       correct = 0
    #        n_examples = 0
    #        for t_x1, t_mask1, t_x2, t_mask2, t_l, t_y in all_test:
    #            correct += sess.run(acc, feed_dict = {d_input:t_x1, q_input:t_x2, y: t_y, l_mask: t_l, training: False})
    #            n_examples += len(t_x1)
    #        test_acc = correct * 100. / n_examples
    #        print('Test Accuracy: %.2f %%' % test_acc)

    print('-'*50)
    print('Start training...')
    
    init = tf.global_variables_initializer()
    train_acc_epoch = []
    train_loss_epoch = []
    dev_acc_epoch = []
    dev_loss_epoch = []
    start_time = time.time()

    with tf.Session() as sess:
        sess.run(init)
        for e in range(num_epoches):
            
            n_updates = 0
            np.random.shuffle(all_train)
            start_time = time.time()
            train_acc_batch = []
            train_loss_batch = []
            dev_acc_batch = []
            dev_loss_batch = []
            
            for idx, (mb_x1, mb_mask1, mb_x2, mb_mask2, mb_l, mb_y) in enumerate(all_train):
                print('Batch Size = %d, # of Examples = %d, max_len = %d' % (mb_x1.shape[0], len(mb_x1), mb_x1.shape[1]))

                y_label = np.zeros((mb_x1.shape[0], num_labels))
 
                for r, i in enumerate(mb_y): # convert (batch) -> (batch, entity_size)
                    y_label[r][i] = 1.

                _, train_loss, train_acc = sess.run([train_op, loss_op,acc1], feed_dict={d_input:mb_x1, q_input:mb_x2, y_1hot: y_label,
                                                                                  answer: np.array(mb_y),
                                                                                  l_mask: mb_l, training: True})
                print('Epoch = %d, Iter = %d (max = %d), Loss = %.2f, Elapsed Time = %.3f (s), Acc = %.2f' %
                                (e, idx, len(all_train), train_loss/mb_x1.shape[0], time.time() - start_time,                                                                                            train_acc*100./mb_x1.shape[0]))
                n_updates += 1
                train_acc_batch.append(train_acc*100./mb_x1.shape[0])
                train_loss_batch.append(train_loss/mb_x1.shape[0])

                if n_updates % eval_iter == 0:
                    saver.save(sess, model_path, global_step=e)
                    correct = 0
                    loss = 0
                    n_examples = 0
                    for d_x1, d_mask1, d_x2, d_mask2, d_l, d_y in all_dev:
                        y_label_dev = np.zeros((d_x1.shape[0], num_labels))
                        for r, i in enumerate(d_y): 
                            y_label_dev[r][i] = 1.
                        loss += sess.run(loss_op, feed_dict = {d_input:d_x1, q_input:d_x2, answer: np.array(d_y),y_1hot: y_label_dev,
                                                               l_mask: d_l, training: False})
                        correct += sess.run(acc, feed_dict = {d_input:d_x1, q_input:d_x2, answer: np.array(d_y),
                                                              l_mask: d_l, training: False})
                        n_examples += len(d_x1)
                    dev_acc = correct * 100./ n_examples
                    dev_loss = loss / n_examples
                    dev_acc_batch.append(dev_acc)
                    dev_loss_batch.append(dev_loss)
                    print('Dev Accuracy: %.2f %%' % dev_acc)
                    if dev_acc > best_acc:
                        best_acc = dev_acc
                        print('Best Dev Accuracy: epoch = %d, n_updates (iter) = %d, acc = %.2f %%' %
                                                   (e, n_updates, dev_acc))
                        
            train_loss_epoch.append(np.mean(train_loss_batch))
            train_acc_epoch.append(np.mean(train_acc_batch))
            dev_loss_epoch.append(np.mean(dev_loss_batch))
            dev_acc_epoch.append(np.mean(dev_acc_batch))
        
        print('-'*50)
        print('Training Finished...')
        print("Model saved in file: %s" % saver.save(sess, model_path))
        return train_loss_epoch, train_acc_epoch, dev_loss_epoch, dev_acc_epoch 
