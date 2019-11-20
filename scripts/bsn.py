#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 yyl     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.
from datetime import datetime
import os, time, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import tensorflow.contrib.slim as slim
from mi_brv.data import *

class BSN(object):
    def __init__(self, dataset):
        self.start_lr = 0.001
        self.train_iters = 5000
        self.datasets = dataset
        self.num_bags = self.datasets.num_bags
        self.dimension = 200
        self.display_step = 50
        self.snapshot = 50
        self.model_save_dir = './model'
        self.batch_size = 1
        self.weight_decay = 0.001
        self.stddev = 0.1

    def model(self, inputs, is_training):
        with slim.arg_scope([slim.fully_connected], 
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=self.stddev),
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            ):
            net = slim.fully_connected(inputs, 256, scope='fc1')
            net = slim.fully_connected(net, 128, scope='fc2')
            net = slim.fully_connected(net, 64, scope='fc3')
            net = slim.dropout(net, keep_prob=0.5, is_training=is_training)
            feas = net
            
            # score pooling
            net = tf.reduce_max(net, 0, keep_dims=True)
            net = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid, scope='fc4')
        return net, feas

    def compute_loss(self, y, logits, is_training):
        cross_entropy = -y * tf.log(logits+1e-7) - (1-y) * tf.log(1-logits+1e-7)
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        
        regularization_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.add_to_collection('losses', regularization_losses)
        return tf.cond(is_training,
                        lambda: tf.add_n(tf.get_collection('losses'), name='total_loss'),
                        lambda: cross_entropy_mean)

    def compute_accuracy(self, y, logits):
        return tf.equal(y, tf.round(logits))

    def run_net1(self, fold=0):
        fea = tf.placeholder(tf.float32, shape=(None, self.dimension), name='fea')
        label = tf.placeholder(tf.float32, name='label')
        
        lr = tf.placeholder(tf.float32, name='lr')
        is_training = tf.placeholder(tf.bool, name='is_training')

        pred, feas = self.model(fea, is_training)
        cost = self.compute_loss(label, pred, is_training)
        accuracy = self.compute_accuracy(label, pred)

        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.MomentumOptimizer(lr, 0.9).minimize(cost, global_step)

        init = tf.initialize_all_variables()
        saver = tf.train.Saver(max_to_keep=50)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)
            best_te_loss = 100.
            step = 1
            while step < self.train_iters:
                if step < 1500:
                    learning_rate = self.start_lr
                elif step < 3000:
                    learning_rate = self.start_lr * 0.5
                else:
                    learning_rate = self.start_lr * 0.25
                batch_fea, batch_label = self.datasets.get_next_batch(step, fold, is_train=True)
                start_time = time.time()
                # load batch_data
                _, batch_loss = sess.run([optimizer, cost], feed_dict={fea:batch_fea, label:batch_label, lr:learning_rate, is_training:True})
                duration = time.time() - start_time
                if step % self.display_step == 0:
                    examples_per_sec = self.batch_size / duration
                    sec_per_batch = float(duration)
                    
                    format_str = ('%s: step = %d, loss = %.5f ( %.1f example/sec; %.3f sec/batch)')
                    print format_str % (datetime.now(), step, batch_loss, examples_per_sec, sec_per_batch)
                        
                if step % self.snapshot == 0 or (step+1) == self.train_iters:
                    # test
                    num_te_batch = len(self.datasets.datasets[fold]['test'])

                    te_loss = np.zeros((num_te_batch,1), dtype=np.float32)
                    te_acc = np.zeros((num_te_batch,1), dtype=np.float32)
                    for idx in range(num_te_batch):
                        te_batch_fea, te_batch_label = self.datasets.get_next_batch(idx, fold, is_train=False)
                        te_loss[idx], te_acc[idx] = sess.run([cost, accuracy], feed_dict={fea:te_batch_fea, label:te_batch_label, lr:0., is_training:False})
                    format_str = ('%s: step = %d, test loss = %.4f, test accuracy = %.4f ')
                    te_loss = np.mean(te_loss)
                    te_acc = np.mean(te_acc)
                    print 'Testing:'
                    print format_str % (datetime.now(), step, te_loss, te_acc)
                    
                    # save model
                    if te_loss < best_te_loss:
                        best_te_loss = te_loss
                        saver.save(sess, self.model_save_dir + '/best_model.ckpt')
                
                step = step + 1
            print 'Optimization Finished!'
            
            # compute test accuracy of best model
            saver.restore(sess, self.model_save_dir + '/best_model.ckpt')
            num_te_batch = len(self.datasets.datasets[fold]['test'])

            te_loss = np.zeros((num_te_batch), dtype=np.float32)
            te_acc = np.zeros((num_te_batch), dtype=np.float32)
            for idx in range(num_te_batch):
                te_batch_fea, te_batch_label = self.datasets.get_next_batch(idx, fold, is_train=False)
                te_loss[idx], te_acc[idx] = sess.run([cost, accuracy], feed_dict={fea:te_batch_fea, label:te_batch_label, lr:0., is_training:False})
            format_str = ('%s: step = %d, test loss = %.4f, test accuracy = %.4f ')
            te_loss = np.mean(te_loss)
            te_acc = np.mean(te_acc)
            print 'Testing:'
            print format_str % (datetime.now(), step, te_loss, te_acc)
            
            # get features of training instances
            num_tr_batch = len(self.datasets.datasets[fold]['train'])
            tr_bags = []
            tr_labels = []
            tr_mask = []
            for idx in range(num_tr_batch):
                tr_batch_fea, tr_batch_label = self.datasets.get_next_batch(idx, fold, is_train=True)
                tr_bags += sess.run([feas], feed_dict={fea:tr_batch_fea, label:batch_label, lr:0., is_training:False})
                tr_labels.append(tr_batch_label)
                tr_mask += [idx for i in range(tr_batch_fea.shape[0])]
            sess.close()
        return te_acc, (tr_bags, tr_labels, tr_mask)

    def model2(self, inputs, tr_bags, tr_mask, num_tr_bags):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=self.stddev),
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                ):
            net = slim.fully_connected(inputs, 256, scope='fc1')
            net = slim.fully_connected(net, 128, scope='fc2')
            net = slim.fully_connected(net, 64, scope='fc3')
            feas = net
            # compute dist between instances
            net = tf.matmul(tr_bags, tf.transpose(net))

            # do max-max pooling to compute bag representation
            net = tf.segment_max(net, tr_mask)
            net = tf.reduce_max(tf.transpose(net), 0, keep_dims=True)
            net = tf.reshape(net, shape=(-1, num_tr_bags))
            
            # classify
            net = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid, scope='fc4')
            return net, feas


    def run_net2(self, tr_data, fold=0):        
        tr_bags, tr_labels, tr_mask = tr_data
        
        fea = tf.placeholder(tf.float32, shape=(None, self.dimension), name='fea')
        label = tf.placeholder(tf.float32, name='label')

        lr = tf.placeholder(tf.float32, name='lr')
        is_training = tf.placeholder(tf.bool, name='is_training')
        
        tr_mask = tf.constant(np.asarray(tr_mask), name='tr_mask')
        tr_bags = tf.constant(np.concatenate(tr_bags), name='tr_bags')

        pred, feas = self.model2(fea, tr_bags, tr_mask, len(tr_labels)) 
        cost = self.compute_loss(label, pred, is_training)
        accuracy = self.compute_accuracy(label, pred)

        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.MomentumOptimizer(lr, 0.9).minimize(cost, global_step)
        
        init = tf.initialize_all_variables()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        saver = tf.train.Saver(max_to_keep=50)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)
            
            step = 1
            best_te_loss = 100.
            if step < 1500:
                learning_rate = self.start_lr
            elif step < 3000:
                learning_rate = self.start_lr * 0.5
            else:
                learning_rate = self.start_lr * 0.25
            while step < self.train_iters:
                batch_fea, batch_label = self.datasets.get_next_batch(step, fold, is_train=True) 
                start_time = time.time()
                # load batch_data
                _, batch_loss, acc = sess.run([optimizer, cost, accuracy], feed_dict={fea:batch_fea, label:batch_label, lr:learning_rate, is_training:True})
                duration = time.time() - start_time
                if step % self.display_step == 0:
                    examples_per_sec = self.batch_size / duration
                    sec_per_batch = float(duration)
                    
                    format_str = ('%s: step = %d, loss = %.5f ( %.1f example/sec; %.3f sec/batch)')
                    print format_str % (datetime.now(), step, batch_loss, examples_per_sec, sec_per_batch)

                if step % self.snapshot == 0 or (step+1) == self.train_iters:
                    # test
                    num_te_batch = len(self.datasets.datasets[fold]['test'])

                    te_loss = np.zeros((num_te_batch), dtype=np.float32)
                    te_acc = np.zeros((num_te_batch), dtype=np.float32)
                    for idx in range(num_te_batch):
                        te_batch_fea, te_batch_label = self.datasets.get_next_batch(idx, fold, is_train=False)
                        te_loss[idx], te_acc[idx] = sess.run([cost, accuracy], feed_dict={fea:te_batch_fea, label:te_batch_label,  lr:0., is_training:False})

                    te_loss = np.mean(te_loss)
                    te_acc = np.mean(te_acc)
                    format_str = ('%s: step = %d, test loss = %.4f, test accuracy = %.4f ')
                    print 'Testing:'
                    print format_str % (datetime.now(), step, te_loss, te_acc)
    
                    # save model
                    if te_loss < best_te_loss:
                        best_te_loss = te_loss
                        saver.save(sess, self.model_save_dir + '/best_model_2.ckpt')
                    
                step = step + 1
            print 'Optimization Finished!'
            
            # compute test accuracy of best model
            saver.restore(sess, self.model_save_dir + '/best_model_2.ckpt')
            num_te_batch = len(self.datasets.datasets[fold]['test'])

            te_loss = np.zeros((num_te_batch), dtype=np.float32)
            te_acc = np.zeros((num_te_batch), dtype=np.float32)
            for idx in range(num_te_batch):
                te_batch_fea, te_batch_label = self.datasets.get_next_batch(idx, fold, is_train=False)
                te_loss[idx], te_acc[idx] = sess.run([cost, accuracy], feed_dict={fea:te_batch_fea, label:te_batch_label, lr:0., is_training:False})
            format_str = ('%s: step = %d, test loss = %.4f, test accuracy = %.4f ')
            te_loss = np.mean(te_loss)
            te_acc = np.mean(te_acc)
            print 'Testing:'
            print format_str % (datetime.now(), step, te_loss, te_acc)
    
            # get features of training instances
            num_tr_batch = len(self.datasets.datasets[fold]['train'])
            tr_bags = []
            tr_labels = []
            tr_mask = []
            for idx in range(num_tr_batch):
                tr_batch_fea, tr_batch_label = self.datasets.get_next_batch(idx, fold, is_train=True)
                tr_bags += sess.run([feas], feed_dict={fea:tr_batch_fea, label:batch_label, lr:0., is_training:False})
                tr_labels.append(tr_batch_label)
                tr_mask += [idx for i in range(tr_batch_fea.shape[0])]
            sess.close()
        return te_acc, (tr_bags, tr_labels, tr_mask)

if __name__ == '__main__':
    dataset_nm = sys.argv[1]
    seed = [22+i*5 for i in range(10)]
    acc = np.zeros((10, 10))
    for idx in range(10):
        dataset = MIL_Dataset(seed[idx], dataset_nm)
        for fold in range(10):
            print 'run=', idx, ' fold=', fold
            net = BSN(dataset)
            _, tr_data = net.run_net1(fold)
            tf.reset_default_graph()
            
            acc[idx][fold], _ = net.run_net2(tr_data, fold)
            tf.reset_default_graph()

    print acc
    print 'acc:'
    print 'mean=', np.mean(acc), ' std=', np.std(acc)
