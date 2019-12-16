import time
from tqdm import *
from glob import glob

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from skip_rnn_cells import *
import sys
import time
from utils import *

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


def DeepLSTM(rnn_type,hidden_size,dropout_rate_embd,dropout_rate_encoder,learning_rate,batch_size,optimizer_name,num_epoches,eval_iter,depth,
                     vocab_size, embd_size, glove_embd_w,num_labels, model_path,
                     all_train, all_dev):
    print('-'* 50)
    print('Creating TF computation graph...')

    if rnn_type == 'lstm':
        print('Using LSTM Cells')
    elif rnn_type == 'gru':
        print('Using GRU Cells')

    train_input = tf.placeholder(dtype=tf.int32, shape=(None, None), name="d_input")
    l_mask = tf.placeholder(dtype=tf.float32, shape=(None, None), name="l_mask") # [batch_size, entity num]
    answer = tf.placeholder(dtype=tf.int32, shape=None, name="label") # batch size vector
    y_1hot= tf.placeholder(dtype=tf.float32, shape=(None, None), name="label_1hot") # onehot encoding of y [batch_size, entitydict]
    training = tf.placeholder(dtype=tf.bool)

    word_embeddings = tf.get_variable("glove", shape=(vocab_size, embd_size), initializer=tf.constant_initializer(glove_embd_w))

    with tf.variable_scope('encoder'): # Encoding Step for Passage (d_ for document)
        embed = tf.nn.embedding_lookup(word_embeddings, train_input) # Apply embeddings: [batch, max passage length in batch, GloVe Dim]
        embed_dropout = tf.layers.dropout(embed, rate=dropout_rate_embd, training=training) # Apply Dropout to embedding layer
        if rnn_type == 'lstm':
            #stack_cell = tf.contrib.cudnn_rnn.CudnnLSTM([hidden_size]*depth)
            #initial_state = stack_cell.trainable_initial_state(batch_size)
            stack_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
                                         tf.nn.rnn_cell.LSTMCell(hidden_size), (1-dropout_rate_encoder)) for n in range(depth)])
        elif rnn_type == 'gru':
            #stack_cell = tf.contrib.cudnn_rnn.CudnnGRU([hidden_size]*depth)
            #initial_state = stack_cell.trainable_initial_state(batch_size)
            stack_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
                                         tf.nn.rnn_cell.GRUCell(hidden_size), (1-dropout_rate_encoder)) for n in range(depth)])

        rnn_outputs, rnn_final_states = tf.nn.dynamic_rnn(stack_cell, embed_dropout, dtype=tf.float32)#initial_state=initial_state)
        batch_states = tf.concat([rnn_final_states[i][1] for i in range(2)],1)  # [batch,depth*h]
        drop_output = tf.layers.dropout(batch_states, rate=dropout_rate_encoder, training=training)

    with tf.variable_scope('dense'): # Prediction Step
        # the final output has dimension [batch, entity#], giving the probabilities of an entity being the answer for examples
        final_prob = tf.layers.dense(drop_output,units=num_labels,use_bias=True) # [batch, entity#]

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
    #    if test_file == None:
    #        return ValueError("No test file specified")
    #    test_examples = utils.load_data(args.test_file)
    #    test_x1, test_x2, test_l, test_y = vectorize(test_examples, word_dict, entity_dict)
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
            
            for idx, (mb_x, mb_mask, mb_l, mb_y) in enumerate(all_train):
                print('Batch Size = %d, # of Examples = %d, max_len = %d' % (mb_x.shape[0], len(mb_x), mb_x.shape[1]))

                y_label = np.zeros((mb_x.shape[0], num_labels))
 
                for r, i in enumerate(mb_y): # convert (batch) -> (batch, entity_size)
                    y_label[r][i] = 1.

                _, train_loss, train_acc = sess.run([train_op, loss_op,acc1], feed_dict={train_input:mb_x, y_1hot: y_label,
                                                                                 answer: np.array(mb_y),
                                                                                 l_mask: mb_l, training: True})
                print('Epoch = %d, Iter = %d (max = %d), Loss = %.2f, Elapsed Time = %.3f (s), Acc = %.2f' %
                                (e, idx, len(all_train), train_loss/mb_x.shape[0], time.time() - start_time,                                                                                            train_acc*100./mb_x.shape[0]))
                n_updates += 1
                train_acc_batch.append(train_acc*100./mb_x.shape[0])
                train_loss_batch.append(train_loss/mb_x.shape[0])

                if n_updates % eval_iter == 0:
                    saver.save(sess, model_path, global_step=e)
                    correct = 0
                    loss = 0
                    n_examples = 0
                    for d_x, d_mask, d_l, d_y in all_dev:
                        y_label_dev = np.zeros((d_x.shape[0], num_labels))
                        for r, i in enumerate(d_y): 
                            y_label_dev[r][i] = 1.
                        loss += sess.run(loss_op, feed_dict = {train_input:d_x, answer: np.array(d_y),y_1hot: y_label_dev,
                                                               l_mask: d_l, training: False})
                        correct += sess.run(acc, feed_dict = {train_input:d_x, answer: np.array(d_y),
                                                              l_mask: d_l, training: False})
                        n_examples += len(d_x)
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
