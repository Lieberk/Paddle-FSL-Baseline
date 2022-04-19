import paddle
import os
import h5py

import configs
from data.datamgr import SimpleDataManager
from io_utils import model_dict, parse_args, get_best_file


def save_features(model, data_loader, outfile):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader) * data_loader.batch_size
    all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
    all_feats = None
    count = 0
    for i, (x, y) in enumerate(data_loader):
        if i % 10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x_var = x
        feats = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list(feats.shape[1:]), dtype='f')
        all_feats[count:count + feats.shape[0]] = feats.cpu().numpy()
        all_labels[count:count + feats.shape[0]] = y.cpu().numpy()
        count = count + feats.shape[0]

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()


if __name__ == '__main__':
    params = parse_args('save_features')

    image_size = 84
    split = params.split
    loadfile = configs.data_dir[params.dataset] + split + '.json'
    checkpoint_dir = '%s/checkpoint/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'

    modelfile = get_best_file(checkpoint_dir)
    outfile = os.path.join(checkpoint_dir.replace("checkpoint", "features"), split + ".hdf5")
    datamgr = SimpleDataManager(image_size, batch_size=64)
    data_loader = datamgr.get_data_loader(loadfile, aug=False)

    model = model_dict[params.model]()
    tmp = paddle.load(modelfile)
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.",
                                 "")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from
            # 'feature.trunk.xx' to 'trunk.xx'
            state[newkey] = state.pop(key)
        else:
            state.pop(key)

    model.load_dict(state)
    model.eval()

    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    save_features(model, data_loader, outfile)
