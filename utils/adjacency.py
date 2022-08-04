import os
import torch
import numpy as np
from scipy import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import rbf_kernel
from torch_geometric.data import Data


def hyperDist(hsi_feat):
    pairwiseDist = torch.cdist(hsi_feat, hsi_feat, p=2)
    return pairwiseDist


def lidarDist(lidar, p=2.0):
    pairwiseDist = torch.cdist(lidar, lidar, p)
    return pairwiseDist


class meta():
    def __init__(self, args, device, ignored_label):
        self.args = args
        self.device = device

        self.hsi = io.loadmat(os.path.join(args.dataset_dir, "HSI.mat"))['HSI'].astype(np.float32)
        self.lidar = io.loadmat(os.path.join(args.dataset_dir, "LiDAR.mat"))['LiDAR'].astype(np.float32)
        if self.lidar.ndim == 3 and args.lidar_bands_num == 1: self.lidar = self.lidar[:,:,0:1]
        if self.lidar.ndim == 2: self.lidar = np.expand_dims(self.lidar, axis=2)
        self.gt = io.loadmat(os.path.join(args.dataset_dir, "gt.mat"))['gt'].astype(np.int64)

        self.TRLabel, self.TSLabel = None, None
        TR_path = os.path.join(args.dataset_dir, "TRLabel.mat")
        TS_path = os.path.join(args.dataset_dir, "TSLabel.mat")
        if os.path.exists(TR_path): self.TRLabel = io.loadmat(TR_path)['TRLabel'].astype(np.int64)
        if os.path.exists(TS_path): self.TSLabel = io.loadmat(TS_path)['TSLabel'].astype(np.int64)

        mask = np.ones(shape=self.gt.shape, dtype=bool)
        mask[self.gt == ignored_label] = False
        x_pos, y_pos = np.nonzero(mask)
        # indices代表非ignored_label点的几何坐标
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        self.labels = torch.tensor([self.gt[i, j] - 1 for i, j in self.indices], dtype=torch.long)
        self.collection = []
        self.collection_pseudo = []

    def pyg_data_generation(self, feat):
        edge_index = self.edge_index_generation()
        data = Data(x=feat, edge_index=edge_index, y=self.labels, train_mask=self.train_mask, test_mask=self.test_mask)
        return data
    
    def mask_generation(self):
        train_mask = torch.zeros(len(self.indices), dtype=torch.bool)
        test_mask = torch.zeros(len(self.indices), dtype=torch.bool)

        if self.TRLabel is not None and self.TSLabel is not None:
            for i, (h, w) in enumerate(self.indices):
                train_mask[i] = True if self.TRLabel[h, w] else False
                test_mask[i] = True if self.TSLabel[h, w] else False
            self.train_mask, self.test_mask = train_mask, test_mask
            return train_mask, test_mask
        
        for i in range(self.labels.max().item() + 1):
            cnt = (self.labels == i).sum()
            candidates = (self.labels == i).nonzero(as_tuple=True)[0]
            self.collection.append(candidates)
            shuffle = torch.randperm(cnt)
            train_idx = shuffle[:self.args.train_samples_per_cls]
            train_mask[candidates[train_idx]] = True
            test_idx = shuffle[self.args.train_samples_per_cls:]
            test_mask[candidates[test_idx]] = True
        self.train_mask, self.test_mask = train_mask, test_mask
        return train_mask, test_mask
    
    def pseudo_labels_generation(self, preds):
        for i in range(preds.max().item() + 1):
            candidates = (preds == i).nonzero(as_tuple=True)[0]
            self.collection_pseudo.append(candidates)
    
    def edge_index_generation(self):
        """
        Method coming from `Hyperspectral Image Classification Based on Deep Attention Graph convolution Network`
        Measurement involved:
        1. Chebyshev distance (Spatial distance)
        2. Spectral information divergence (HSI)
        3. Kernel spectral angle mapper (HSI)
        Form: d_{mix}(h_i, h_j) = log(d_{ij}) \times d_{SID}(h_i || h_j) \times sin(d_{KSAM}(h_i, j_j))
        """
        # Preprocessing
        hsi = np.empty(shape=(len(self.indices), self.hsi.shape[-1]), dtype=self.hsi.dtype)
        lidar = np.empty(shape=(len(self.indices), self.lidar.shape[-1]), dtype=self.lidar.dtype)
        for idx, (i, j) in enumerate(self.indices):
            hsi[idx, :], lidar[idx, :] = self.hsi[i, j], self.lidar[i, j]
        hsi, lidar = MinMaxScaler().fit_transform(hsi), MinMaxScaler().fit_transform(lidar)

        # Hyperspectral distance.
        ## 1. spectral information divergence (SID)
        distribution = torch.from_numpy(hsi).softmax(dim=1) # 原文中的归一化方式是除以行和。这里为了后面的kl散度计算过程中不涉及到0值，从而使用softmax函数。
        log_distribution = torch.from_numpy(hsi).log_softmax(dim=1)
        kl_divergence = (distribution * log_distribution).sum(dim=1, keepdim=True) - torch.einsum("ik, jk -> ij", distribution, log_distribution) # Reference (https://discuss.pytorch.org/t/calculate-p-pair-wise-kl-divergence/131424)
        epsilon = 3e-8 # Reference [pytorch/pytorch, issue #8069](https://github.com/pytorch/pytorch/issues/8069)
        kl_divergence.clamp_(min=epsilon)
        Distance = kl_divergence.add_(kl_divergence.clone().T)

        ## 2. (kernel) spectral angle mapper (SAM)
        epsilon = 3e-8
        if self.args.use_ksam:
            gamma = 1. / (2 * self.args.delta_ksam**2)
            phi = torch.from_numpy(rbf_kernel(hsi, hsi, gamma)).clamp_(min=epsilon-1, max=1-epsilon)
        else:
            hsi = torch.from_numpy(hsi)
            hsi /= hsi.norm(p=2, dim=1, keepdim=True).clamp_(min=epsilon)
            phi = (hsi@hsi.T).clamp_(min=epsilon-1, max=1-epsilon)
        Distance.mul_(phi.acos_().sin_())

        # Distance = hyperDist(torch.from_numpy(hsi))
        # Lidar distance.
        if self.args.mul_lidar:
            Distance.mul_(lidarDist(torch.from_numpy(lidar)))
        else:
            Distance.add_(self.args.lambd * lidarDist(torch.from_numpy(lidar), self.args.p))

        # Spatial distance constraint.
        ## If the spatial smoothness is emphasized, then pixels with a chebyshev distance of `1` will definitely be considered similar.
        ## This will result in all eight neighbors aroung each pixel being difinitely determined as similar.
        indices_tensor = torch.from_numpy(self.indices).float()
        chebyshevDist = torch.cdist(indices_tensor, indices_tensor, p=float("inf"))
        spatialDist = chebyshevDist.log2_() if self.args.emphasize_smooth else chebyshevDist.add_(1).log2_()
        spatialDist.clamp_(min=epsilon)

        if self.args.mul_spat:
            Distance.mul_(spatialDist)
        else:
            Distance.add_(self.args.beta * spatialDist)

        # # assert (Distance >= 0).all(), "===> Non-positive distance exists."
        _, indices_dist = torch.topk(Distance, k=self.args.neighbors_num, largest=False)
        edge_index = torch.empty(size=(2, len(self.indices) * self.args.neighbors_num), dtype=torch.long)
        for idx, (i, j) in enumerate(self.indices):
            start = idx * self.args.neighbors_num
            end = start + self.args.neighbors_num

            src = torch.empty(size=(self.args.neighbors_num, ), dtype=torch.long).fill_(idx)
            dst = indices_dist[idx]

            edge_index[:, start: end] = torch.stack((dst, src), dim=0)
        return edge_index
