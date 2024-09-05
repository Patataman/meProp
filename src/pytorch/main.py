'''
Train MLPs for MNIST using meProp
'''
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from argparse import ArgumentParser

import torch

from data import get_mnist
from util import TestGroup


def get_args():
    # a simple use example (not unified)
    parser = ArgumentParser()
    parser.add_argument(
        '--n_epoch', type=int, default=20, help='number of training epochs')
    parser.add_argument(
        '--d_hidden', type=int, default=512, help='dimension of hidden layers')
    parser.add_argument(
        '--n_layer',
        type=int,
        default=3,
        help='number of layers, including the output layer')
    parser.add_argument(
        '--d_minibatch', type=int, default=32, help='size of minibatches')
    parser.add_argument(
        '--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument(
        '--k',
        type=int,
        default=30,
        help='k in meProp (if invalid, e.g. 0, do not use meProp)')
    parser.add_argument(
        '--unified',
        dest='unified',
        action='store_true',
        help='use unified meProp')
    parser.add_argument(
        '--no-unified',
        dest='unified',
        action='store_false',
        help='do not use unified meProp')
    parser.add_argument(
        '--random_seed', type=int, default=12976, help='random seed')
    parser.set_defaults(unified=False)
    return parser.parse_args()


def get_args_unified():
    # a simple use example (unified)
    parser = ArgumentParser()
    parser.add_argument(
        '--n_epoch', type=int, default=20, help='number of training epochs')
    parser.add_argument(
        '--d_hidden', type=int, default=500, help='dimension of hidden layers')
    parser.add_argument(
        '--n_layer',
        type=int,
        default=3,
        help='number of layers, including the output layer')
    parser.add_argument(
        '--d_minibatch', type=int, default=50, help='size of minibatches')
    parser.add_argument(
        '--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument(
        '--k',
        type=int,
        default=30,
        help='k in meProp (if invalid, e.g. 0, do not use meProp)')
    parser.add_argument(
        '--unified',
        dest='unified',
        action='store_true',
        help='use unified meProp')
    parser.add_argument(
        '--no-unified',
        dest='unified',
        action='store_false',
        help='do not use unified meProp')
    parser.add_argument(
        '--random_seed', type=int, default=12976, help='random seed')
    parser.set_defaults(unified=True)
    return parser.parse_args()


def main():
    args = get_args()
    trn, dev, tst = get_mnist()

    group = TestGroup(
        args,
        trn,
        args.d_minibatch,
        args.d_hidden,
        args.n_layer,
        args.dropout,
        args.unified,
        dev,
        tst,
        file=sys.stdout)

    # results may be different at each run
    group.run(0, args.n_epoch)
    # mbsize: 32, hidden size: 512, layer: 3, dropout: 0.1, k: 0
    # 0：dev set: Average loss: 0.0943, Accuracy: 4861/5000 (97.22%)
    #     test set: Average loss: 0.0952, Accuracy: 9694/10000 (96.94%)
    # 1：dev set: Average loss: 0.0940, Accuracy: 4861/5000 (97.22%)
    # 2：dev set: Average loss: 0.0839, Accuracy: 4872/5000 (97.44%)
    #     test set: Average loss: 0.0876, Accuracy: 9742/10000 (97.42%)
    # 3：dev set: Average loss: 0.0870, Accuracy: 4881/5000 (97.62%)
    #     test set: Average loss: 0.0830, Accuracy: 9749/10000 (97.49%)
    # 4：dev set: Average loss: 0.0799, Accuracy: 4893/5000 (97.86%)
    #     test set: Average loss: 0.0870, Accuracy: 9766/10000 (97.66%)
    # 5：dev set: Average loss: 0.0813, Accuracy: 4887/5000 (97.74%)
    # 6：dev set: Average loss: 0.0808, Accuracy: 4906/5000 (98.12%)
    #     test set: Average loss: 0.0802, Accuracy: 9794/10000 (97.94%)
    # 7：dev set: Average loss: 0.0937, Accuracy: 4900/5000 (98.00%)
    # 8：dev set: Average loss: 0.0889, Accuracy: 4901/5000 (98.02%)
    # 9：dev set: Average loss: 0.0944, Accuracy: 4899/5000 (97.98%)
    # 10：dev set: Average loss: 0.0970, Accuracy: 4909/5000 (98.18%)
    #     test set: Average loss: 0.0901, Accuracy: 9833/10000 (98.33%)
    # 11：dev set: Average loss: 0.1040, Accuracy: 4901/5000 (98.02%)
    # 12：dev set: Average loss: 0.0912, Accuracy: 4912/5000 (98.24%)
    #     test set: Average loss: 0.1005, Accuracy: 9807/10000 (98.07%)
    # 13：dev set: Average loss: 0.1134, Accuracy: 4897/5000 (97.94%)
    # 14：dev set: Average loss: 0.0927, Accuracy: 4911/5000 (98.22%)
    # 15：dev set: Average loss: 0.1171, Accuracy: 4904/5000 (98.08%)
    # 16：dev set: Average loss: 0.1191, Accuracy: 4914/5000 (98.28%)
    #     test set: Average loss: 0.1144, Accuracy: 9823/10000 (98.23%)
    # 17：dev set: Average loss: 0.1449, Accuracy: 4892/5000 (97.84%)
    # 18：dev set: Average loss: 0.1299, Accuracy: 4902/5000 (98.04%)
    # 19：dev set: Average loss: 0.1193, Accuracy: 4909/5000 (98.18%)
    # $98.28|98.23 at 16
    group.run(args.k, args.n_epoch)
    # mbsize: 32, hidden size: 512, layer: 3, dropout: 0.1, k: 30
    # 0：dev set: Average loss: 0.2653, Accuracy: 4776/5000 (95.52%)
    #     test set: Average loss: 0.2730, Accuracy: 9520/10000 (95.20%)
    # 1：dev set: Average loss: 0.1361, Accuracy: 4846/5000 (96.92%)
    #     test set: Average loss: 0.1615, Accuracy: 9650/10000 (96.50%)
    # 2：dev set: Average loss: 0.1089, Accuracy: 4873/5000 (97.46%)
    #     test set: Average loss: 0.1151, Accuracy: 9714/10000 (97.14%)
    # 3：dev set: Average loss: 0.1002, Accuracy: 4866/5000 (97.32%)
    # 4：dev set: Average loss: 0.1127, Accuracy: 4859/5000 (97.18%)
    # 5：dev set: Average loss: 0.0875, Accuracy: 4879/5000 (97.58%)
    #     test set: Average loss: 0.0934, Accuracy: 9758/10000 (97.58%)
    # 6：dev set: Average loss: 0.0896, Accuracy: 4886/5000 (97.72%)
    #     test set: Average loss: 0.0817, Accuracy: 9775/10000 (97.75%)
    # 7：dev set: Average loss: 0.0910, Accuracy: 4895/5000 (97.90%)
    #     test set: Average loss: 0.0957, Accuracy: 9771/10000 (97.71%)
    # 8：dev set: Average loss: 0.0870, Accuracy: 4909/5000 (98.18%)
    #     test set: Average loss: 0.0805, Accuracy: 9802/10000 (98.02%)
    # 9：dev set: Average loss: 0.0777, Accuracy: 4911/5000 (98.22%)
    #     test set: Average loss: 0.0794, Accuracy: 9804/10000 (98.04%)
    # 10：dev set: Average loss: 0.0828, Accuracy: 4900/5000 (98.00%)
    # 11：dev set: Average loss: 0.0956, Accuracy: 4904/5000 (98.08%)
    # 12：dev set: Average loss: 0.0819, Accuracy: 4910/5000 (98.20%)
    # 13：dev set: Average loss: 0.0896, Accuracy: 4913/5000 (98.26%)
    #     test set: Average loss: 0.0969, Accuracy: 9802/10000 (98.02%)
    # 14：dev set: Average loss: 0.0907, Accuracy: 4914/5000 (98.28%)
    #     test set: Average loss: 0.1006, Accuracy: 9796/10000 (97.96%)
    # 15：dev set: Average loss: 0.1111, Accuracy: 4905/5000 (98.10%)
    # 16：dev set: Average loss: 0.1524, Accuracy: 4888/5000 (97.76%)
    # 17：dev set: Average loss: 0.1202, Accuracy: 4912/5000 (98.24%)
    # 18：dev set: Average loss: 0.1145, Accuracy: 4904/5000 (98.08%)
    # 19：dev set: Average loss: 0.1303, Accuracy: 4910/5000 (98.20%)
    # $98.28|97.96 at 14


def main_unified():
    args = get_args_unified()
    trn, dev, tst = get_mnist()

    # change the sys.stdout to a file object to write the results to the file
    group = TestGroup(
        args,
        trn,
        args.d_minibatch,
        args.d_hidden,
        args.n_layer,
        args.dropout,
        args.unified,
        dev,
        tst,
        file=sys.stdout)

    # results may be different at each run
    group.run(0)
    # mbsize: 50, hidden size: 500, layer: 3, dropout: 0.1, k: 0
    # 0：dev set: Average loss: 0.1066, Accuracy: 4832/5000 (96.64%)
    #     test set: Average loss: 0.1114, Accuracy: 9646/10000 (96.46%)
    # 1：dev set: Average loss: 0.0917, Accuracy: 4861/5000 (97.22%)
    #     test set: Average loss: 0.0974, Accuracy: 9721/10000 (97.21%)
    # 2：dev set: Average loss: 0.0842, Accuracy: 4879/5000 (97.58%)
    #     test set: Average loss: 0.0860, Accuracy: 9740/10000 (97.40%)
    # 3：dev set: Average loss: 0.0778, Accuracy: 4897/5000 (97.94%)
    #     test set: Average loss: 0.0783, Accuracy: 9776/10000 (97.76%)
    # 4：dev set: Average loss: 0.0902, Accuracy: 4883/5000 (97.66%)
    # 5：dev set: Average loss: 0.0727, Accuracy: 4898/5000 (97.96%)
    #     test set: Average loss: 0.0679, Accuracy: 9818/10000 (98.18%)
    # 6：dev set: Average loss: 0.0797, Accuracy: 4905/5000 (98.10%)
    #     test set: Average loss: 0.0861, Accuracy: 9790/10000 (97.90%)
    # 7：dev set: Average loss: 0.0853, Accuracy: 4907/5000 (98.14%)
    #     test set: Average loss: 0.0842, Accuracy: 9792/10000 (97.92%)
    # 8：dev set: Average loss: 0.0832, Accuracy: 4910/5000 (98.20%)
    #     test set: Average loss: 0.0938, Accuracy: 9771/10000 (97.71%)
    # 9：dev set: Average loss: 0.1054, Accuracy: 4898/5000 (97.96%)
    # 10：dev set: Average loss: 0.0853, Accuracy: 4905/5000 (98.10%)
    # 11：dev set: Average loss: 0.0811, Accuracy: 4914/5000 (98.28%)
    #     test set: Average loss: 0.0802, Accuracy: 9822/10000 (98.22%)
    # 12：dev set: Average loss: 0.0884, Accuracy: 4917/5000 (98.34%)
    #     test set: Average loss: 0.0884, Accuracy: 9813/10000 (98.13%)
    # 13：dev set: Average loss: 0.0829, Accuracy: 4923/5000 (98.46%)
    #     test set: Average loss: 0.0864, Accuracy: 9820/10000 (98.20%)
    # 14：dev set: Average loss: 0.0954, Accuracy: 4906/5000 (98.12%)
    # 15：dev set: Average loss: 0.0998, Accuracy: 4912/5000 (98.24%)
    # 16：dev set: Average loss: 0.1189, Accuracy: 4905/5000 (98.10%)
    # 17：dev set: Average loss: 0.1121, Accuracy: 4901/5000 (98.02%)
    # 18：dev set: Average loss: 0.1180, Accuracy: 4913/5000 (98.26%)
    # 19：dev set: Average loss: 0.1266, Accuracy: 4904/5000 (98.08%)
    # $98.46|98.20 at 13
    group.run()
    # mbsize: 50, hidden size: 500, layer: 3, dropout: 0.1, k: 30
    # 0：dev set: Average loss: 0.1731, Accuracy: 4753/5000 (95.06%)
    #     test set: Average loss: 0.1845, Accuracy: 9455/10000 (94.55%)
    # 1：dev set: Average loss: 0.1399, Accuracy: 4799/5000 (95.98%)
    #     test set: Average loss: 0.1441, Accuracy: 9590/10000 (95.90%)
    # 2：dev set: Average loss: 0.1093, Accuracy: 4830/5000 (96.60%)
    #     test set: Average loss: 0.1111, Accuracy: 9653/10000 (96.53%)
    # 3：dev set: Average loss: 0.0950, Accuracy: 4856/5000 (97.12%)
    #     test set: Average loss: 0.0994, Accuracy: 9682/10000 (96.82%)
    # 4：dev set: Average loss: 0.0882, Accuracy: 4873/5000 (97.46%)
    #     test set: Average loss: 0.0861, Accuracy: 9747/10000 (97.47%)
    # 5：dev set: Average loss: 0.0766, Accuracy: 4884/5000 (97.68%)
    #     test set: Average loss: 0.0829, Accuracy: 9744/10000 (97.44%)
    # 6：dev set: Average loss: 0.0862, Accuracy: 4877/5000 (97.54%)
    # 7：dev set: Average loss: 0.0816, Accuracy: 4888/5000 (97.76%)
    #     test set: Average loss: 0.0804, Accuracy: 9753/10000 (97.53%)
    # 8：dev set: Average loss: 0.0760, Accuracy: 4894/5000 (97.88%)
    #     test set: Average loss: 0.0798, Accuracy: 9774/10000 (97.74%)
    # 9：dev set: Average loss: 0.0904, Accuracy: 4874/5000 (97.48%)
    # 10：dev set: Average loss: 0.0813, Accuracy: 4892/5000 (97.84%)
    # 11：dev set: Average loss: 0.0705, Accuracy: 4904/5000 (98.08%)
    #     test set: Average loss: 0.0758, Accuracy: 9772/10000 (97.72%)
    # 12：dev set: Average loss: 0.0718, Accuracy: 4904/5000 (98.08%)
    # 13：dev set: Average loss: 0.0734, Accuracy: 4897/5000 (97.94%)
    # 14：dev set: Average loss: 0.0791, Accuracy: 4885/5000 (97.70%)
    # 15：dev set: Average loss: 0.0804, Accuracy: 4896/5000 (97.92%)
    # 16：dev set: Average loss: 0.0826, Accuracy: 4895/5000 (97.90%)
    # 17：dev set: Average loss: 0.0873, Accuracy: 4891/5000 (97.82%)
    # 18：dev set: Average loss: 0.0856, Accuracy: 4898/5000 (97.96%)
    # 19：dev set: Average loss: 0.0887, Accuracy: 4906/5000 (98.12%)
    #     test set: Average loss: 0.0909, Accuracy: 9785/10000 (97.85%)
    # $98.12|97.85 at 19


if __name__ == '__main__':
    # uncomment to run meprop
    # main()
    # run unified meprop
    main_unified()
