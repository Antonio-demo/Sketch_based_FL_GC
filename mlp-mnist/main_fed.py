# -*- coding: utf-8 -*-
# Python version: 3.6
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, FFD, k_svd
from models.test import test_img
import time



if __name__ == '__main__':
    start = time.perf_counter()
    # parse(分析) args
    args = args_parser()
    #args.device表示分配可选设备GPU或者CPU
    args.device = torch.device('cuda:{}'.format(args.gpu)
                               if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        # transforms.compose串联多个图片，即[transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]
        # 是列表，遍历
        # transforms.ToTensor()是一个将图片转换为形状为(C,H,W)的tensor格式
        # transforms.Normalize((0.1307,), (0.3081,))是将tensor形式归一化，范围在0~1.0之间
        trans_mnist = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
        # 将数据集MNIST分割为训练集和测试集
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        # args.iid即args的独立同分布
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)

    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
            dict_users = cifar_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')

    img_size = dataset_train[0][0].shape

    # build model
    #cnn即卷积神经网络
    # 将CNNMnist模型放到设备上，该模型是全局模型
    #net_glob = CNNMnist(args=args).to(args.device)
    #mlp即多层感知器
    """
    """
    if args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    #print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    #print(w_glob)

    layer_input_weight = w_glob['layer_input.weight']
    #print(layer_input_weight.shape)

    # t = layer_input_weight.size()
    # print(t)

    # number = layer_input_weight.numel()
    # print(number)
    # 200行784列的矩阵，总共有156800个元素
    array_layer_input_weight = layer_input_weight.cpu().numpy()
    # raw_avg是行平均向量的矩阵，总共784个
    raw_avg = np.mean(array_layer_input_weight, axis=0)
    g1 = array_layer_input_weight[0, :] - raw_avg
    # print(g1)
    g1 = g1.reshape(16,147)

    b = [[0.00000000] * 147] * 20
    B = np.matrix(b)
    result = FFD(g1, B)
    # print(result.shape)
    # 这个就是我最终想要的，接下来放入全局模型中
    x = k_svd(result, 15)
    tensor_x = torch.from_numpy(x)
    # print(tensor_x)
    # print(tensor_x.shape)

    w_glob['layer_input.weight'] = tensor_x
    # print(w_glob['layer_input.weight'].shape)
    # training
    loss_train1 = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        #args.frac * args.num_users表示选中的用户的数量
        m = max(int(args.frac * args.num_users), 1)
        #np.random.choice(range(args.num_users), m, replace=False)表示从所有的用户中随机选取m个用户
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights即更新全局模型中的权重
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob,上传到全局模型中
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train1.append(loss_avg)

        # testing
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)



    """
    
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    """


    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    end = time.perf_counter()
    print("Running time:%s seconds"%(end-start))


