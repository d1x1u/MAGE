import os

import numpy as np
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
import torch
from torch import autocast, nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from .extractor import SSARN
from .LARSSGD import LARS
from .mlp_head import MLPHead
from ..utils.data import Dataset_finetune, TrainDataset, TrainPipeline
from ..utils.report import compute_metrics, metrics2text
from ..utils.toolbox import get_timestamp


class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = SSARN(args)
        self.projector = MLPHead(args.hsi_channels[-1], args.mlp_hidden_size, args.projection_size, args.use_swish)

    def forward(self, x):
        feature = self.encoder(x)
        projection = self.projector(feature)
        return feature, projection


class BYOLTrainer:
    def __init__(self, args, device, labels_text, ignored_label):
        self.online_network = Network(args).to(device)
        self.predictor = MLPHead(args.projection_size, args.mlp_hidden_size, args.projection_size, args.use_swish).to(device)
        self.target_network = Network(args).to(device)
        self.args = args
        self.device = device
        self.labels_text = labels_text
        self.ignored_label = ignored_label
        self.class_num = args.class_num
        self.epochs = args.epochs_cluster
        self.lr_cluster = args.lr_cluster
        self.lr_cluster_min = args.lr_cluster_min
        self.warmup_epochs = args.warmup_epochs
        self.base_m = args.momentum
        self.model_path_cluster = args.model_path_cluster
        self.best_model_path_cluster = args.best_model_path_cluster

        # train_dataset = TrainDataset(args)
        # self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
        # self.train_transform = get_train_transforms(args.patch_size).to(device)
        train_pipeline = TrainPipeline(batch_size=args.batch_size, num_threads=args.num_threads, device_id=0, seed=args.seed, args=args)
        train_pipeline.build()
        if args.fastmode:
            self.train_loader = DALIClassificationIterator(pipelines=train_pipeline, auto_reset=True, size=args.num_labels, last_batch_policy=LastBatchPolicy.DROP, last_batch_padded=True)
        else:
            self.train_loader = DALIClassificationIterator(pipelines=train_pipeline, auto_reset=True, size=args.H*args.W, last_batch_policy=LastBatchPolicy.DROP, last_batch_padded=True)

        self.steps = 0
        self.total_steps = self.epochs * len(self.train_loader) // args.batch_size
        self.warmup_steps = self.warmup_epochs * len(self.train_loader) // args.batch_size
        self.optimizer = LARS(
            params=self.collect_params([self.online_network, self.predictor]),
            lr=args.lr_cluster, momentum=0.9, weight_decay=1.0e-6
        )

        self.initializes_target_network()
    
    @torch.no_grad()
    def collect_params(self, model_list, exclude_bias_and_bn=True):
        """
        exclude_bias_and bn: exclude bias and bn from both weight decay and LARS adaptation
            in the PyTorch implementation of ResNet, `downsample.1` are bn layers
        """
        param_list = []
        for model in model_list:
            for name, param in model.named_parameters():
                if exclude_bias_and_bn and ('bn' in name or 'downsample.1' in name or 'bias' in name):
                    param_dict = {'params': param, 'weight_decay': 0., 'lars_exclude': True}
                else:
                    param_dict = {'params': param}
                param_list.append(param_dict)
        return param_list
    
    @torch.no_grad()
    def _update_momentum(self, step):
        self.m = 1 - (1 - self.base_m) * (np.cos(np.pi * step / self.total_steps) + 1) / 2

    @torch.no_grad()
    def _update_learning_rate(self, step):
        max_lr = self.lr_cluster
        min_lr = self.lr_cluster_min
        if step < self.warmup_steps:
            lr = (max_lr - min_lr) * step / self.warmup_steps + min_lr
        else:
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos((step - self.warmup_steps) * np.pi / self.total_steps))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @torch.no_grad()
    def _update_target_network_parameters(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = self.m * param_k.data + (1. - self.m) * param_q.data
    
    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)
    
    @torch.no_grad()
    def initializes_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def update(self, batch_view_1, batch_view_2):
        # use kornia to do augmentation first.
        # batch_view_1 = self.train_transform(batch_view_1)
        # batch_view_2 = self.train_transform(batch_view_2)

        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1)[-1])
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2)[-1])

        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)[-1]
            targets_to_view_1 = self.target_network(batch_view_2)[-1]
        
        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()
    
    def train(self):
        timestamp = get_timestamp()

        self.online_network.train()
        scaler = GradScaler()
        for epoch in range(self.epochs):
            LOSS = 0
            # loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False)
            # for batch, ((batch_view_1, batch_view_2), _) in loop:
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False)
            for batch, data in loop:
            # for data in self.train_loader:
                loop.set_description(f'Epoch [{epoch+1}/{self.epochs}]')

                self._update_learning_rate(self.steps)
                self._update_momentum(self.steps)
                self.steps += 1

                batch_view_1 = data[0]['data']
                batch_view_2 = data[0]['label']

                # batch_view_1 = batch_view_1.to(self.device, non_blocking=True)
                # batch_view_2 = batch_view_2.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                with autocast():
                    loss = self.update(batch_view_1, batch_view_2)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                # loss.backward()
                # self.optimizer.step()
                self._update_target_network_parameters()
                
                loop.set_postfix(loss=loss.item())
                LOSS += loss.item() / len(self.train_loader)
            print(f"Epoch [{epoch + 1}/{self.epochs}]\t Loss: {LOSS}")
            wandb.log({"loss":LOSS})

        self.save_path = os.path.join(self.model_path_cluster, timestamp)
        self.save()
    
    def finetune(self, args, train_mask, test_mask, linear_protocol=True):
        checkpoint = torch.load(os.path.join(self.save_path, "checkpoint.tar"))
        self.online_network.load_state_dict(checkpoint['online_network'])

        train_dataset = Dataset_finetune(args, self.ignored_label, train_mask)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size_finetune, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
        test_dataset = Dataset_finetune(args, self.ignored_label, test_mask)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
        
        classifier = nn.Linear(in_features=args.hsi_channels[-1], out_features=self.class_num, bias=True).to(self.device)
        criterion = nn.CrossEntropyLoss()

        if linear_protocol:
            self.online_network.eval()
            for p in self.online_network.parameters():
                p.requires_grad = False
            optimizer = Adam(params=classifier.parameters(), lr=args.lr_finetune_linear, weight_decay=args.weight_decay_finetune)
            epochs = args.epochs_finetune_linear
        else:
            self.online_network.train()
            optimizer = Adam(
                params=[
                    {'params': self.online_network.parameters(), 'lr': args.lr_finetune_full},
                    {'params': classifier.parameters(), 'lr': args.lr_finetune_multiplier*args.lr_finetune_full}
                ],
                weight_decay=args.weight_decay_finetune
            )
            epochs = args.epochs_finetune_full

        for epoch in range(epochs):
            LOSS = 0
            train_loop = tqdm(train_loader, total=len(train_loader), leave=False)
            for x, y in train_loop:
            # for x, y in train_loader:
                train_loop.set_description(f'Epoch [{epoch+1}/{epochs}]')

                x, y = x.to(self.device), y.to(self.device)
                feature, _ = self.online_network(x)
                out = classifier(feature)
                loss = criterion(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loop.set_postfix(loss=loss.item())
                LOSS += loss.item() / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}]\t Loss: {LOSS}")
            if linear_protocol: wandb.log({"loss_finetune_linear": LOSS})
            else: wandb.log({"loss_finetune_full": LOSS})

        """Finetune Inference"""
        self.online_network.eval()
        with torch.no_grad():
            test_loop = tqdm(test_loader, total=len(test_loader), leave=False)
            preds, labels = [], []
            for x, y in test_loop:
                x, y = x.to(self.device), y.to(self.device)
                feature, _ = self.online_network(x)
                out = classifier(feature)
                preds.extend(torch.argmax(out, dim=1).detach().cpu().numpy())
                labels.extend(y.detach().cpu().numpy())
            prediction, target = np.array(preds), np.array(labels)
            n_classes, ignored_labels = int(target.max()), [self.ignored_label]
            results = compute_metrics(prediction, target, n_classes, ignored_labels)
            # oa, aa, kappa = results['OA'], results['AA'], results['Kappa']
            oa, aa, kappa, pa = results['OA'], results['AA'], results['Kappa'], results['PA']
            # metrics2text(results, self.labels_text)

        text = ""
        text += f"Overall Accuracy: {oa:.04f}\n"
        text += "---\n"
        text += f"Average Accuracy: {aa:.04f}\n"
        text += "---\n"
        text += f"Kappa: {kappa:.04f}\n"
        text += "---\n"
        print(text)

        if linear_protocol: wandb.log({"OA_finetune_linear": oa, "AA_finetune_linear": aa, "Kappa_finetune_linear": kappa})
        else: wandb.log({"OA_finetune_full": oa, "AA_finetune_full": aa, "Kappa_finetune_full": kappa})
        # return oa, aa, kappa
        return oa, aa, kappa, pa, prediction

    @torch.no_grad()
    def inference(self, args, mask):
        test_dataset = Dataset_finetune(args, self.ignored_label, mask)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
        loop = tqdm(test_loader, total=len(test_loader), leave=False)

        self.online_network.eval()
        features, projections, labels = None, None, None
        for x, y in loop:
            loop.set_description("Inference")

            x, y = x.to(self.device), y.to(self.device)
            
            feature, projection = self.online_network(x)
            features = torch.cat((features, feature), dim=0) if features is not None else feature
            projections = torch.cat((projections, projection), dim=0) if projections is not None else projection
            labels = torch.cat((labels, y), dim=0) if labels is not None else y

        return {'feats': features.detach().cpu(), 'projs': projections.detach().cpu(), 'labels': labels.detach().cpu()}

    def save(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        state = {
            'online_network': self.online_network.state_dict(),
            'predictor': self.predictor.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, os.path.join(self.save_path, "checkpoint.tar"))

    def reload(self):
        """
        Use `reload` function when training the BYOLTrainer and GNN sub-networks seperately,
        for accelerating the iteration speed.
        The funciton requests that the user puts the trained BYOLTrainer model in the folder before reloading.

        Note: Maybe the joint training is necessary.
        """
        model_id = os.listdir(self.best_model_path_cluster)[0]
        model_fp = os.path.join(self.best_model_path_cluster, model_id)

        checkpoint = torch.load(model_fp)
        self.online_network.load_state_dict(checkpoint['online_network'])
        self.predictor.load_state_dict(checkpoint['predictor'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        