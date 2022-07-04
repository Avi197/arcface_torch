import mxnet as mx
from mxnet import ndarray as nd
import numpy as np
import argparse
import pickle
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'eval'))

parser = argparse.ArgumentParser(description='Package LFW images')
# general
parser.add_argument('--data-dir', default='', help='')
parser.add_argument('--image-size', type=str, default='112,96', help='')
parser.add_argument('--output', default='', help='path to save.')


# lfw_data = nd.empty((len(lfw_paths), 3, image_size[0], image_size[1]))

def get_paths(lfw_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(
                lfw_dir, pair[0],
                pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(
                lfw_dir, pair[0],
                pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(
                lfw_dir, pair[0],
                pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(
                lfw_dir, pair[2],
                pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(
                path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            print('not exists', path0, path1)
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


if __name__ == '__main__':
    args = parser.parse_args()
    lfw_dir = args.data_dir
    image_size = [int(x) for x in args.image_size.split(',')]
    lfw_pairs = read_pairs(os.path.join(lfw_dir, 'pairs.txt'))
    lfw_paths, issame_list = get_paths(lfw_dir, lfw_pairs, 'jpg')
    lfw_bins = []

    i = 0
    for path in lfw_paths:
        with open(path, 'rb') as fin:
            _bin = fin.read()
            lfw_bins.append(_bin)
            # img = mx.image.imdecode(_bin)
            # img = nd.transpose(img, axes=(2, 0, 1))
            # lfw_data[i][:] = img
            i += 1
            if i % 1000 == 0:
                print('loading lfw', i)

    with open(args.output, 'wb') as f:
        pickle.dump((lfw_bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
