import os
import time
from scipy import io
import numpy as np
from numpy import ndarray
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from nvidia.dali import pipeline_def, fn


def get_dataset_info(dataset_name):
    if dataset_name == "MUUFL":
        labels_text = ['Trees', 'Grass ground', 'Mixed ground', 'Dirt and sand', 'Road', 'Water', 'Buildings', 'Shadow', 'Sidewalk', 'Yellow curb', 'Cloth panels']
        ignored_label = -1
    elif dataset_name == "Trento":
        labels_text = ['Apple trees', 'Buildings', 'Ground', 'Woods', 'Vineyard', 'Roads']
        ignored_label = -1
    elif dataset_name == "Houston":
        labels_text = ['Health grass', 'Stressed grass', 'Synthetic grass', 'Trees', 'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway', 'Railway', 'Parking lot 1', 'Parking lot 2', 'Tennis court', 'Running track']
        ignored_label = -1
    else:
        raise ValueError("Dataset must be one of MUUFL, Trento, Houston")
    return labels_text, ignored_label


def load_processed_dataset(args):
    hsi = io.loadmat(os.path.join(args.dataset_dir, "HSI.mat"))['HSI'].astype(np.float32)
    H, W, C = hsi.shape
    hsi = hsi.reshape(-1, C)
    if args.use_pca:
        # Reference: [Importance of Feature Scaling](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html)
        hsi = StandardScaler().fit_transform(hsi)
        hsi = PCA(n_components=args.pca_component).fit_transform(hsi)
    else:
        hsi = MinMaxScaler().fit_transform(hsi)
    hsi = hsi.reshape(H, W, -1)

    lidar = io.loadmat(os.path.join(args.dataset_dir, "LiDAR.mat"))['LiDAR'].astype(np.float32)
    if lidar.ndim == 3 and args.lidar_bands_num == 1: lidar = lidar[:,:,0:1]
    if lidar.ndim == 2: lidar = np.expand_dims(lidar, axis=2)
    H, W, C = lidar.shape
    lidar = lidar.reshape(-1, C)
    lidar = MinMaxScaler().fit_transform(lidar)
    lidar = lidar.reshape(H, W, -1)

    gt = io.loadmat(os.path.join(args.dataset_dir, "gt.mat"))['gt'].astype(np.int64)

    return hsi, lidar, gt


def sliding_window(HSI, LiDAR, gt, step=1, window_size=(9, 9), fastmode=False, finetune=False, ignored_label=-1):
    """Sliding window generator over an input image.

    Args:
        HSI: (H, W, C), hyperspectral image
        LiDAR: (H, W, C), LiDAR image
        gt: (H, W), the ground truth
        step: int stride of the sliding window
        window_size: input tuple, height and width of the window
        finetune: option for adjusting sliding window's behavior.
        ignored_label: assist the `finetune` option.
    Returns:
        hsi_patches, lidar_patches, labels
    """

    hsi_patches, lidar_patches, labels = [], [], []

    # 对数据进行镜像填充，纳入图像边缘处的样本。
    h, w = window_size
    HSI = np.pad(HSI, ((h//2, h//2), (w//2, w//2), (0, 0)), mode="reflect")
    LiDAR = np.pad(LiDAR, ((h//2, h//2), (w//2, w//2), (0, 0)), mode="reflect")
    gt = np.pad(gt, ((h//2, h//2), (w//2, w//2)), mode="constant", constant_values=0)
    H, W = HSI.shape[:2]

    indices = []
    for y in range(0, H - h + 1):
        for x in range(0, W - w + 1):
            if fastmode == True and gt[y+h//2, x+w//2] == ignored_label:
                continue
            if finetune == True and gt[y+h//2, x+w//2] == ignored_label:  
                continue
            else:
                hsi_patches.append(HSI[y:y+h, x:x+w])
                lidar_patches.append(LiDAR[y:y+h, x:x+w])
                labels.append(gt[y+h//2, x+w//2])
                indices.append((y, x))

    return np.array(hsi_patches), np.array(lidar_patches), np.array(labels), np.array(indices)


@pipeline_def
def TrainPipeline(args):
    data_root = os.path.join(args.dali_file_root, "pca", str(args.patch_size)) if args.use_pca else os.path.join(args.dali_file_root, "no_pca", str(args.patch_size))
    if not os.path.exists(data_root): 
        os.makedirs(data_root)
        npy_generation(args)
    data_files = sorted(os.listdir(data_root))
    view1 = fn.readers.numpy(device='gpu', file_root=data_root, files=data_files, shuffle_after_epoch=True, seed=args.seed, prefetch_queue_depth=2)
    view1 = fn.rotate(view1, angle=fn.random.uniform(values=[0., 90., 180., 270.]), fill_value=0)
    view1 = fn.random_resized_crop(view1, size=args.patch_size)
    view1 = fn.flip(view1, horizontal=fn.random.coin_flip(probability=0.5), vertical=0)
    view1 = fn.flip(view1, horizontal=0, vertical=fn.random.coin_flip(probability=0.5))
    view1 = fn.erase(
        view1, axis_names="HW", normalized_anchor=True, normalized_shape=True, fill_value=0,
        anchor=fn.random.uniform(range=(0., 1.), shape=(2*(args.patch_size**2 // 20), )),
        shape=fn.random.uniform(range=(0.12, 0.2), shape=(2*(args.patch_size**2 // 20), ))
    )
    view1 = fn.transpose(view1, perm=[2, 0, 1])

    view2 = fn.readers.numpy(device='gpu', file_root=data_root, files=data_files, shuffle_after_epoch=True, seed=args.seed+1, prefetch_queue_depth=2)
    view2 = fn.rotate(view2, angle=fn.random.uniform(values=[0., 90., 180., 270.]), fill_value=0)
    view2 = fn.random_resized_crop(view2, size=args.patch_size)
    view2 = fn.flip(view2, horizontal=fn.random.coin_flip(probability=0.5), vertical=0)
    view2 = fn.flip(view2, horizontal=0, vertical=fn.random.coin_flip(probability=0.5))
    view2 = fn.erase(
        view2, axis_names="HW", normalized_anchor=True, normalized_shape=True, fill_value=0,
        anchor=fn.random.uniform(range=(0., 1.), shape=(2*(args.patch_size**2 // 20), )),
        shape=fn.random.uniform(range=(0.12, 0.2), shape=(2*(args.patch_size**2 // 20), ))
    )
    view2 = fn.transpose(view2, perm=[2, 0, 1])

    return view1, view2


def npy_generation(args):
    hsi, lidar, gt = load_processed_dataset(args)
    hsi_patches, lidar_patches, _, indices = sliding_window(hsi, lidar, gt, step=args.step, window_size=(args.patch_size, args.patch_size), fastmode=args.fastmode)
    concat_patches = np.concatenate((hsi_patches, lidar_patches), axis=-1)

    data_root = os.path.join(args.dali_file_root, "pca", str(args.patch_size)) if args.use_pca else os.path.join(args.dali_file_root, "no_pca", str(args.patch_size))
    for idx, (y, x) in enumerate(indices):
        filename = 'y'+str(y).zfill(4)+'_'+'x'+str(x).zfill(4)+'.npy'
        data_path = os.path.join(data_root, filename)
        np.save(file=data_path, arr=concat_patches[idx])


class MyDataset(Dataset):
    def __init__(self, args, data_transform=None, finetune=False, ignored_label=None):
        super().__init__()
        self.data_transform = data_transform
        hsi, lidar, gt = load_processed_dataset(args)
        self.hsi_patches, self.lidar_patches, self.labels, _ = sliding_window(hsi, lidar, gt, step=args.step, window_size=(args.patch_size, args.patch_size), finetune=finetune, ignored_label=ignored_label)
        self.hsi_patches = torch.tensor(self.hsi_patches).permute(0, 3, 1, 2)
        self.lidar_patches = torch.tensor(self.lidar_patches).permute(0, 3, 1, 2)
        self.labels = torch.tensor(self.labels)
    
    def __len__(self):
        return len(self.hsi_patches)


class TrainDataset(MyDataset):
    def __init__(self, args):
        super().__init__(args)
    def __getitem__(self, idx):
        hsi_patch = self.hsi_patches[idx]
        lidar_patch = self.lidar_patches[idx]
        concat_patch = torch.cat((hsi_patch, lidar_patch), dim=0)
        label = self.labels[idx]
        return (concat_patch, concat_patch), label


class TestDataset(MyDataset):
    def __init__(self, args):
        super().__init__(args)
    def __getitem__(self, idx):
        hsi_patch = self.hsi_patches[idx]
        lidar_patch = self.lidar_patches[idx]
        concat_patch = torch.cat((hsi_patch, lidar_patch), dim=0)
        label = self.labels[idx]
        return concat_patch, label


class Dataset_finetune(MyDataset):
    def __init__(self, args, ignored_label, mask):
        super().__init__(args, finetune=True, ignored_label=ignored_label)
        self.hsi_patches = self.hsi_patches[mask]
        self.lidar_patches = self.lidar_patches[mask]
        self.labels = self.labels[mask] - 1

    def __len__(self):
        return len(self.hsi_patches)

    def __getitem__(self, idx):
        hsi_patch = self.hsi_patches[idx]
        lidar_patch = self.lidar_patches[idx]
        concat_patch = torch.cat((hsi_patch, lidar_patch), dim=0)
        label = self.labels[idx]
        return concat_patch, label


def visualize_dataset(x: ndarray, y: ndarray, dataset: str):
    ncls = np.max(y) + 1
    ylim = [0, ncls-1]

    base_cmp = 'gist_rainbow'
    base = plt.cm.get_cmap(base_cmp)
    color_list = base(np.linspace(0, 1, ncls))
    cmap_name = base.name + str(ncls)

    start = time.time()
    tsne = TSNE().fit_transform(x)
    print(f'elapsed time {time.time() - start}s')

    plt.scatter(tsne[:, 0], tsne[:, 1], c=y, s=5, cmap=base.from_list(cmap_name, color_list, ncls))
    cbar = plt.colorbar(ticks=np.arange(ylim[1] + 1))
    plt.clim(-0.5, ncls-0.5)
    cbar.ax.set_yticklabels(np.arange(0, ncls, 1))
    plt.title(f"Visualizing {dataset}'s HSI through t-SNE", fontsize=15)
    plt.savefig(f"{dataset}_HSI_tSNE.png", dpi=300)
    plt.show()
