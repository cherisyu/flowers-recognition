# -*- coding: UTF-8 -*-
# fully connected network

import tensorflow as tf
import numpy as np
import load_data as ld
import random
import time

class FC():
    def __init__(self):
        pass

    def fcn(self,x,reg_rate,is_training):
        '''fully connected network'''
        regularizer = tf.contrib.layers.l2_regularizer(scale=reg_rate)
        shape = x.get_shape()
        in_dim = shape[1]*shape[2]*shape[3]
        x_flatten = tf.reshape(x,(-1,in_dim))
        #layer_1
        with tf.variable_scope('fc1',regularizer=regularizer):
            weight_1 = tf.get_variable(name = 'weight1',shape=[in_dim,1024],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        biases_1 = tf.get_variable(name = 'biases1',shape=[1024],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        fc_1 = tf.nn.leaky_relu(tf.matmul(x_flatten,weight_1)+biases_1,name='fc1')
        drop_1 = tf.layers.dropout(inputs=fc_1,rate=0.5,training=is_training,name='drop_1')
        #layer_2
        with tf.variable_scope('fc2',regularizer=regularizer):
            weight_2 = tf.get_variable(name = 'weight2',shape=[1024,512],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        biases_2 = tf.get_variable(name = 'biases2',shape=[512],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        fc_2 = tf.nn.leaky_relu(tf.matmul(drop_1,weight_2)+biases_2,name='fc2')
        drop_2 = tf.layers.dropout(inputs=fc_2,rate=0.5,training=is_training,name='drop_2')

        #layer_3
        with tf.variable_scope('fc3',regularizer=regularizer):
            weight_3 = tf.get_variable(name = 'weight3',shape=[512,256],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        biases_3 = tf.get_variable(name = 'biases3',shape=[256],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        fc_3 = tf.nn.leaky_relu(tf.matmul(drop_2,weight_3)+biases_3,name='fc3')
        drop_3 = tf.layers.dropout(inputs=fc_3,rate=0.5,training=is_training,name='drop_3')

        #layer_4
        with tf.variable_scope('fc4',regularizer=regularizer):
            weight_4 = tf.get_variable(name = 'weight4',shape=[256,128],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        biases_4 = tf.get_variable(name = 'biases4',shape=[128],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        fc_4 = tf.nn.leaky_relu(tf.matmul(drop_3,weight_4)+biases_4,name='fc4')
        drop_4 = tf.layers.dropout(inputs=fc_4,rate=0.5,training=is_training,name='drop_4')

        with tf.variable_scope('fc5',regularizer=regularizer):
            weight_5 = tf.get_variable(name = 'weight5',shape=[128,5],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        biases_5 = tf.get_variable(name = 'biases5',shape=[5],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        predicted = tf.matmul(drop_4,weight_5)+biases_5

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg = tf.reduce_sum(reg_losses)

        return predicted,reg

def run_fc(lr,epochs,batch_size,reg_rate):
    #数据有问题
    F = ld.Flower()
    flowers,Image,Label,Label_onehot = F.read_img()
    Train_img,Train_label,Validation_img,Validation_label,Test_img,Test_label = F.split_data(flowers,Image,Label,Label_onehot,returnwhat=1)
    # Train_img = np.random.randn(200,128,128,3)
    # Train_label = np.random.uniform(0,5,200).astype(np.int32)

    N = Train_img.shape[0]
    M = Validation_img.shape[0]
    T = Test_img.shape[0]
    index = list(range(N))

    model = FC()
    img = tf.placeholder(dtype=tf.float32,shape=(None,128,128,3))
    true = tf.placeholder(dtype=tf.int64,shape=(None))
    is_training = tf.placeholder(tf.bool)
    predicted_onehot,reg = model.fcn(img,reg_rate,is_training)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true,logits=predicted_onehot))+reg
    predicted = tf.argmax(predicted_onehot,axis=-1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,true),tf.float32))
    optim = tf.train.AdamOptimizer(lr).minimize(loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    start = time.time()
    for i in range(epochs):
        batch_id = random.sample(index,batch_size)
        train_img_batch = Train_img[batch_id]
        train_label_batch = Train_label[batch_id]

        _,loss_,accuracy_ = sess.run([optim,loss,accuracy],feed_dict={
            img:train_img_batch,
            true:train_label_batch,
            is_training:True
            })
        if (i+1)%1000==0:
            val_accuracy = []
            # print('step:',i+1,'loss:',loss_,'accuracy:',accuracy_)
            s = 0
            while(s<M):
                e = min(s+batch_size,M)
                val_acc = sess.run(accuracy,feed_dict={
                        img:Validation_img[s:min(s+batch_size,M)],
                        true:Validation_label[s:min(s+batch_size,M)],
                        is_training:False
                        })
                val_accuracy.append(val_acc*(e-s))
                s = e
            val_acc = sum(val_accuracy)/M
            end = time.time()
            duration = end - start
            start = time.time()
            print('step {:d} \t loss = {:.3f} \t train_accuracy =  {:.3f} \t val_accuracy = {:.3f} \t ({:.3f} sec/1000_step)'.format(i+1,loss_,accuracy_,val_acc,duration))

    t = 0
    test_accuracy = []
    while(t<T):
        e = min(t+batch_size,T)
        test_acc = sess.run(accuracy,feed_dict={
            img:Test_img[t:e],
            true:Test_label[t:e],
            is_training:False
            })
        test_accuracy.append(test_acc*(e-t))
        t = e
    print('test accuracy is',sum(test_accuracy)/T)