import os
import argparse
import yaml
import random
import torch
import numpy as np
from scipy import io
from datetime import datetime, timedelta, timezone
from utils.data import get_dataset_info, visualize_dataset


def initialize(dataset_name):
    labels_text, ignored_label = get_dataset_info(dataset_name)

    parser = argparse.ArgumentParser()
    config = yaml_config_hook(f"config/{dataset_name}.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    device = get_device(ordinal=0)
    seed_everything(args.seed)

    return args, device, labels_text, ignored_label


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


def get_device(ordinal=0):
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device


def seed_everything(seed: int):
    r"""
        deterministic, possibly at the cost of reduced performance
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def get_timestamp():
    """URL: https://blog.csdn.net/weixin_39715012/article/details/121048110
    """
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )

    # 协调世界时
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    # 北京时间
    beijing_now = utc_now.astimezone(SHA_TZ)

    return beijing_now.strftime("%Y_%m_%d_%H_%M_%S")


def save_model(model_path, timestamp, model, optimizer, current_epoch):
    out_path = os.path.join(model_path, timestamp)
    out = os.path.join(out_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    torch.save(state, out)


if __name__ == "__main__":
    r"""Test the function `visualize_dataset`
    Request:
    1. There is a GPU Accelerated t-SNE version on Github.
    2. The qualitative colormap for t-SNE can be better, for some colors are hard to distinguish.
    """
    folder = '../dataset/MUUFL/raw'
    name = 'muufl_gulfport_campus_1_hsi_220_label.mat'
    path = os.path.join(folder, name)

    data = io.loadmat(path)
    hsi = data['hsi']['Data'][0][0]
    gt = data['hsi'][0][0]['sceneLabels'][0][0]['labels']

    mask = np.ones(shape=gt.shape, dtype=bool)
    ignored_label = -1
    mask[gt == ignored_label] = False
    x_pos, y_pos = np.nonzero(mask)
    indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])

    x = np.empty(shape=(len(indices), hsi.shape[2]), dtype=hsi.dtype)
    for idx, (i, j) in enumerate(indices):
        x[idx, :] = hsi[i, j]

    labels = np.array([gt[i, j] - 1 for i, j in indices], dtype=np.int64)

    visualize_dataset(x, labels, 'MUUFL')
