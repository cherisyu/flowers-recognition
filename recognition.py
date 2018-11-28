# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import vgg
import fc
import resnet
import resnet_pretrained as res
import svm
import knn
import time
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--batch_size',default=64,type=int,help='number of batch size')
    parser.add_argument('--epochs',default=10000,type=int,help='number of training epochs')
    parser.add_argument('--lr',default=0.0001,type=float,help='learning rate')
    parser.add_argument('--reg',default=0.005,type=float,help='regularizer rate')
    parser.add_argument('--method',required=True,choices=[
        'vgg',
        'svm',
        'knn',
        'fc',
        'resnet34',
        'resnet50'],help='The model will be used')
    args = parser.parse_args()
    return args

def main(args):
    '''
    flowers recognition gogogo!
    '''
    if args.method=='vgg':
        print('Using vgg network for flowers recognition')
        vgg.run_vgg(args.lr,args.epochs,args.batch_size,args.reg)

    if args.method=='fc':
        print('Using fully connected network for flowers recognition')
        fc.run_fc(args.lr,args.epochs,args.batch_size,args.reg)

    if args.method=='resnet34':
        print('Using deep residual network(34-layers) for flowers recognition')
        resnet.run_resnet(args.lr,args.epochs,args.batch_size,args.reg)

    if args.method=='resnet50':
        print('Using deep residual network(50-layers) pretrained for flowers recognition')
        res.run_resnet50(args.batch_size,args.epochs,args.lr)

    if args.method=='svm':
        print('Using Support Vector Machine for flowers recognition')
        svm.run_svm()

    if args.method=='knn':
        print('Using K nearest neighbors for flowers recognition')
        knn.run_knn()

if __name__ == '__main__':
    main(parse_args())

