# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import random
import time
import load_data as ld

def model(x,is_training,reg_rate):
    '''vgg model'''

    regularizer = tf.contrib.layers.l2_regularizer(scale=reg_rate)

    conv1_1 = tf.layers.conv2d(inputs=x,filters=64,kernel_size=3,strides=1,padding='SAME',activation=tf.nn.relu,name='conv1_1')
    pool_1 = tf.nn.max_pool(conv1_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool_1')
    drop_1 = tf.layers.dropout(inputs=pool_1,rate=0.5,training=is_training,name='drop_1')

    conv2_1 = tf.layers.conv2d(inputs=drop_1,filters=128,kernel_size=3,strides=1,padding='SAME',activation=tf.nn.relu,name='conv2_1')
    pool_2 = tf.nn.max_pool(conv2_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool_2')
    drop_2 = tf.layers.dropout(inputs=pool_2,rate=0.5,training=is_training,name='drop_2')

    conv3_1 = tf.layers.conv2d(inputs=drop_2,filters=256,kernel_size=3,strides=1,padding='SAME',activation=tf.nn.relu,name='conv3_1')
    conv3_2 = tf.layers.conv2d(inputs=conv3_1,filters=256,kernel_size=3,strides=1,padding='SAME',activation=tf.nn.relu,name='conv3_2')
    pool_3 = tf.nn.max_pool(conv3_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool_3')
    drop_3 = tf.layers.dropout(inputs=pool_3,rate=0.5,training=is_training,name='drop_3')

    conv4_1 = tf.layers.conv2d(inputs=drop_3,filters=512,kernel_size=3,strides=1,padding='SAME',activation=tf.nn.relu,name='conv4_1')
    conv4_2 = tf.layers.conv2d(inputs=conv4_1,filters=512,kernel_size=3,strides=1,padding='SAME',activation=tf.nn.relu,name='conv4_2')
    pool_4 = tf.nn.max_pool(conv4_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool_4')
    drop_4 = tf.layers.dropout(inputs=pool_4,rate=0.5,training=is_training,name='drop_4')

    conv5_1 = tf.layers.conv2d(inputs=drop_4,filters=512,kernel_size=3,strides=1,padding='SAME',activation=tf.nn.relu,name='conv5_1')
    conv5_2 = tf.layers.conv2d(inputs=conv5_1,filters=512,kernel_size=3,strides=1,padding='SAME',activation=tf.nn.relu,name='conv5_2')
    pool_5 = tf.nn.max_pool(conv5_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool_5')
    drop_5 = tf.layers.dropout(inputs=pool_5,rate=0.5,training=is_training,name='drop_5')


    conv6_1 = tf.layers.conv2d(inputs=drop_5,filters=128,kernel_size=1,strides=1,padding='SAME',activation=tf.nn.relu,name='conv6_1')
    drop_6 = tf.layers.dropout(inputs=conv6_1,rate=0.5,training=is_training,name='drop_6')

    shape = drop_6.get_shape()
    dim = shape[1]*shape[2]*shape[3]
    drop_flatten = tf.reshape(drop_6,[-1,dim])

    with tf.variable_scope('fc',regularizer=regularizer):
        weight_1 = tf.Variable(tf.zeros(shape=[dim,5]),name='weight')
    biases_1 = tf.Variable(tf.zeros(shape=[5]),name='biaes')
    predicted = tf.matmul(drop_flatten,weight_1)+biases_1

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg = tf.reduce_sum(reg_losses)

    return predicted,reg

def run_vgg(lr,epochs,batch_size,reg_rate):
    '''
    lr:learning rate
    epochs: training epochs
    batch_size: number of training samples of a batch
    '''
    F = ld.Flower()
    flowers,Image,Label,Label_onehot = F.read_img()
    Train_img,Train_label,Validation_img,Validation_label,Test_img,Test_label = F.split_data(flowers,Image,Label,Label_onehot,1)

    N = Train_img.shape[0]
    index = list(range(N))
    M = Validation_img.shape[0]
    T = Test_img.shape[0]

    img = tf.placeholder(dtype=tf.float32,shape=(None,128,128,3))
    true = tf.placeholder(dtype=tf.int64,shape=(None))
    is_training = tf.placeholder(tf.bool)

    predicted_onehot,reg = model(img,is_training,reg_rate)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true,logits=predicted_onehot))+reg
    predicted = tf.argmax(tf.nn.softmax(predicted_onehot),axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,true),tf.float32))
    optim = tf.train.AdamOptimizer(lr).minimize(loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    # index = list(range(200))
    start = time.time()
    for i in range(epochs):
        batch_id = random.sample(index,batch_size)
        train_img_batch = Train_img[batch_id]
        train_label_batch = Train_label[batch_id]

        _,loss_,accuracy_= sess.run([optim,loss,accuracy],feed_dict={
            img:train_img_batch,
            true:train_label_batch,
            is_training:True
            })
        if (i+1)%200==0:
            val_accuracy = []
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
            print('step {:d} \t loss = {:.3f} \t train_accuracy =  {:.3f} \t val_accuracy = {:.3f} \t ({:.3f} sec/200_step)'.format(i+1,loss_,accuracy_,val_acc,duration))
            # print(pred)
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

