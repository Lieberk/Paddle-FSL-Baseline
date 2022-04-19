import paddle
import numpy as np
import os
import random
import time

import configs
import data.feature_loader as feat_loader
from methods.baselinefinetune import BaselineFinetune
from io_utils import model_dict, parse_args


def feature_evaluation(cl_data_file, model, n_way=5, n_support=5, n_query=15):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])  # stack each batch

    z_all = paddle.to_tensor(np.array(z_all))

    model.n_query = n_query
    scores = model.set_forward(z_all, is_feature=True)
    pred = scores.cpu().numpy().argmax(axis=1)
    y = np.repeat(range(n_way), n_query)
    acc = np.mean(pred == y) * 100
    return acc


if __name__ == '__main__':
    params = parse_args('test')

    acc_all = []
    iter_num = 600
    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)

    if params.method == 'baseline':
        model = BaselineFinetune(model_dict[params.model], **few_shot_params)
    elif params.method == 'baseline++':
        model = BaselineFinetune(model_dict[params.model], loss_type='dist', **few_shot_params)
    else:
        raise ValueError('Unknown method')

    checkpoint_dir = '%s/checkpoint/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'

    split_str = params.split
    novel_file = os.path.join(checkpoint_dir.replace("checkpoint", "features"),
                              split_str + ".hdf5")  # defaut split = novel, but you can also test base or val classes
    cl_data_file = feat_loader.init_loader(novel_file)

    for i in range(iter_num):
        acc = feature_evaluation(cl_data_file, model, n_query=15, **few_shot_params)
        acc_all.append(acc)

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

    with open('./record/results.txt', 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        aug_str = '-aug' if params.train_aug else ''
        exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' % (
        params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.test_n_way)
        acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num))
        f.write('Time: %s, Setting: %s, Acc: %s \n' % (timestamp, exp_setting, acc_str))
