import numpy as np
import paddle
import paddle.optimizer as optim
import os

import configs
from data.datamgr import SimpleDataManager
from methods.baselinetrain import BaselineTrain
from io_utils import model_dict, get_resume_file, parse_args, get_logger


def train(base_loader, val_loader, logger, model, optimization, start_epoch, stop_epoch, params):
    if optimization == 'Adam':
        optimizer = optim.Adam(parameters=model.parameters())
    else:
        raise ValueError('Unknown optimization, please define by yourself')
    max_acc = 0
    for epoch in range(start_epoch, stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader, logger, optimizer)  # model are called by reference, no need to return
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = model.test_loop(val_loader)
        if acc > max_acc:
            # with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            paddle.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            paddle.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

    return model


if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args('train')
    base_file = configs.data_dir[params.dataset] + 'base.json'
    val_file = configs.data_dir[params.dataset] + 'val.json'
    image_size = 84
    optimization = 'Adam'
    params.stop_epoch = 400  # default

    if params.method in ['baseline', 'baseline++']:
        base_datamgr = SimpleDataManager(image_size, batch_size=64)
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
        val_datamgr = SimpleDataManager(image_size, batch_size=64)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)

        if params.method == 'baseline':
            model = BaselineTrain(model_dict[params.model], params.num_classes)
        elif params.method == 'baseline++':
            model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='dist')
    else:
        raise ValueError('Unknown method')

    params.checkpoint_dir = '%s/checkpoint/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = paddle.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_dict(tmp['state'])

    logger = get_logger(
        filename=os.path.join(params.log_path, 'log.txt'),
        logger_name='master_logger')
    model = train(base_loader, val_loader, logger, model, optimization, start_epoch, stop_epoch, params)
