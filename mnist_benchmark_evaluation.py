# Acknowledgement: The code is based on the DataDAM implementation from the following repository:
# https://github.com/DataDistillation/DataDAM/tree/main
import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
from ptflops import get_model_complexity_info

def calculate_flops(model, input_size):
    macs, params = get_model_complexity_info(model, (1, input_size, input_size), as_strings=True, print_per_layer_stat=False, verbose=False)
    print(f"Model FLOPs: {macs}, Params: {params}")

def main():
    # Parameters setting for mnist train
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    # Dataset changed to MNIST
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    # Same model for all experiments
    parser.add_argument('--model', type=str, default='ConvNetD3', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    # Number of experiments changed to 1 for less time
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=3, help='the number of evaluating randomly initialized models')
    # Epochs changed to 20
    parser.add_argument('--epoch_eval_train', type=int, default=20, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    # Initialization synthetic images from real images
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='./mnist_data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='./mnist_result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    
    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

        ''' training '''
        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s training begins'%get_time())

        for model_eval in model_eval_pool:
            print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s'%(args.model, model_eval))
            # Removed the overwritting of the epoch_eval_train parameter, just use the defined value
            args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc)
            accs = []
            for it_eval in range(args.num_eval):
                net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, images_all, labels_all, testloader, args)
                accs.append(acc_test)
            print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))

    calculate_flops(net_eval, (1, im_size, im_size))

if __name__ == '__main__':
    main()