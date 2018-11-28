# -*- coding: UTF-8 -*-
'''resnet_34 with dropout'''
import numpy as np
import tensorflow as tf
import random
import time
import load_data as ld

class Resnet():
    def __init__(self):
        pass

    def conv_bn_re(self,inputs,input_dim,out_dim,strides=(1,1),use_relu=True,use_bn=True,k=3,is_training=True,scope='conv_bn_re'):
        with tf.variable_scope(scope):
            x=tf.layers.conv2d(inputs=inputs,filters=out_dim,kernel_size=k,strides=strides,padding='same')
            if use_bn:
                x=tf.contrib.layers.batch_norm(x,is_training=is_training)
            if use_relu:
                x=tf.nn.relu(x)
            return x

    def residual(self,inputs,input_dim,out_dim,k=3,strides=(1,1),is_training=True,scope='residual'):
        with tf.variable_scope(scope):
            assert inputs.get_shape().as_list()[3]==input_dim
            #low layer 3*3>3*3
            x=self.conv_bn_re(inputs,input_dim,out_dim,strides=strides,is_training=is_training,scope='up_1')
            x=self.conv_bn_re(x,out_dim,out_dim,use_relu=False,is_training=is_training,scope='up_2')
            #skip,up layer 1*1
            skip=self.conv_bn_re(inputs,input_dim,out_dim,strides=strides,use_relu=False,k=1,is_training=is_training,scope='low')
            #skip+x
            res=tf.nn.relu(tf.add(skip,x))
            return res

    def res_block(self,inputs,input_dim,out_dim,n,k=3,is_training=True,scope='res_block'):
        with tf.variable_scope(scope):
            x=self.residual(inputs,input_dim,out_dim,k=k,is_training=is_training,scope='residual_0')
            x = tf.layers.dropout(inputs=x,rate=0.5,training=is_training,name='drop_0')
            for i in range(1,n):
                x=self.residual(x,out_dim,out_dim,k=k,is_training=is_training,scope='residual_%d'%i)
                x = tf.layers.dropout(inputs=x,rate=0.5,training=is_training,name='drop_%d'%i)
            return x

    def resnet(self,x,is_training,reg_rate):
        '''resnet-34'''
        regularizer = tf.contrib.layers.l2_regularizer(scale=reg_rate)

        conv1 = tf.layers.conv2d(inputs=x,filters=64,kernel_size=7,strides=2,padding='same',name='conv1')
        pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=3,strides=2,padding='same',name='maxpool')

        conv2 = self.res_block(inputs=pool1,input_dim=64,out_dim=64,n=3,is_training=is_training,scope='conv2')

        conv3 = self.res_block(inputs=conv2,input_dim=64,out_dim=128,n=3,is_training=is_training,scope='conv3')

        conv4 = self.res_block(inputs=conv3,input_dim=128,out_dim=256,n=3,is_training=is_training,scope='conv4')

        conv5 = self.res_block(inputs=conv4,input_dim=256,out_dim=512,n=3,is_training=is_training,scope='conv5')

        pool2 = tf.layers.average_pooling2d(conv5,pool_size=1,strides=1,padding='same',name='pool2')

        shape = pool2.get_shape()
        dim = shape[1]*shape[2]*shape[3]
        pool_flatten = tf.reshape(pool2,[-1,dim])
        fc = tf.contrib.layers.fully_connected(inputs=pool_flatten,num_outputs=5,activation_fn=None,weights_regularizer=regularizer,scope='fc')

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg = tf.reduce_sum(reg_losses)

        return fc,reg

def run_resnet(lr,epochs,batch_size,reg_rate):
    '''
    lr: learning rate
    epochs: number of trainings
    batch_size: number of samples of a batch
    '''
    F = ld.Flower()
    flowers,Image,Label,Label_onehot = F.read_img()
    Train_img,Train_label,Validation_img,Validation_label,Test_img,Test_label = F.split_data(flowers,Image,Label,Label_onehot,1,train=0.85,val=0.9)


    N = Train_img.shape[0]
    index = list(range(N))
    M = Validation_img.shape[0]
    T = Test_img.shape[0]

    model = Resnet()
    img = tf.placeholder(dtype=tf.float32,shape=(None,128,128,3))
    true = tf.placeholder(dtype=tf.int64,shape=(None))
    is_training = tf.placeholder(tf.bool)

    # steps=tf.Variable(0,name='global_step',trainable=False)
    # lr=tf.train.exponential_decay(lr,steps,100,0.95,staircase=True,name= 'learning_rate')
    # update=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    predicted_onehot,reg = model.resnet(img,is_training,reg_rate)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true,logits=predicted_onehot))+reg
    predicted = tf.argmax(tf.nn.softmax(predicted_onehot),axis=1)
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

        # sess.run(update)
        _,loss_,accuracy_= sess.run([optim,loss,accuracy],feed_dict={
            img:train_img_batch,
            true:train_label_batch,
            is_training:True
            })
        if (i+1)%100==0:
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
            print('step {:d} \t loss = {:.3f} \t train_accuracy =  {:.3f} \t val_accuracy = {:.3f} \t ({:.3f} sec/100_step)'.format(i+1,loss_,accuracy_,val_acc,duration))
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












