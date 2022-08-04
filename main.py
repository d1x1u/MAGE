# -*- coding: utf-8 -*-
import math
import os

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR

from model.byol_trainer import BYOLTrainer
from model.gnn import Gnn
from utils.adjacency import meta
from utils.report import cm_viz, compute_metrics, metrics2text, show_statistics
from utils.toolbox import initialize, get_timestamp


if __name__ == "__main__":
    dataset_name = "MUUFL" # "MUUFL" / "Trento" / "Houston"
    args, device, labels_text, ignored_label = initialize(dataset_name)

    # ----------------------------------------------------------------
    # 1. Train the SFEM module.
    sfem = BYOLTrainer(args, device, labels_text, ignored_label)
    if args.reload_cluster and os.listdir(args.best_model_path_cluster):
        sfem.reload()
    else:
        sfem.train()
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # 2. train the SGTM module.
    statistics = []
    timestamp = get_timestamp()
    gen = meta(args, device, ignored_label)
    train_mask, test_mask = gen.mask_generation()
    feat = None
    for x, y in gen.indices:
        hsi = gen.hsi[x, y]
        lidar = gen.lidar[x, y]
        concat = np.concatenate((hsi, lidar), axis=0)
        feat = concat if feat is None else np.concatenate((feat, concat), axis=0)
    feat = torch.from_numpy(feat.reshape(args.num_labels, -1))
    # feat = sfem.inference(args, train_mask+test_mask)['feats]
    data = gen.pyg_data_generation(feat).to(device)

    for rep in range(args.num_replicates):
        model = Gnn(args, num_features=data.x.shape[1], num_classes=data.y.max().item() + 1).to(device)

        ce_criterion = torch.nn.CrossEntropyLoss() #TODO: 考虑带权交叉熵，应对类别不平衡问题。这一点在其他的数据集上可以考虑使用。
        optimizer = Adam(model.parameters(), lr=args.lr_gnn, weight_decay=args.weight_decay_gnn)
        # 看看scheduler是否能让optimizer变得更好，想起之前sheduler对Adam的curse
        lr_lambda = lambda epoch: epoch / args.warmup_epochs_gnn if epoch < args.warmup_epochs_gnn else (args.lr_cluster_min + 0.5*(args.lr_gnn-args.lr_cluster_min)*(1.0+math.cos((epoch-args.warmup_epochs)/(args.epochs_gnn-args.warmup_epochs_gnn)*math.pi)))
        scheduler = LambdaLR(optimizer, [lr_lambda]) if args.use_warmup_gnn else ExponentialLR(optimizer, gamma=0.999)
       
        model_dir = os.path.join(args.model_path_gnn, timestamp)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, f"model{rep}.pth")

        best_aa, best_oa, best_kappa = 0, 0, 0
        for epoch in range(1, args.epochs_gnn + 1):
            model.train()
            out = model(data)
            optimizer.zero_grad()
            loss = ce_criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                model.eval()
                pred = model(data).argmax(dim=1)
                results = compute_metrics(
                    prediction=pred.detach().cpu().numpy(), #pred[data.test_mask].detach().cpu().numpy(),
                    target=data.y.detach().cpu().numpy(), #data.y[data.test_mask].detach().cpu().numpy(),
                    n_classes=int(data.y.max()),
                    ignored_labels=[ignored_label]
                )
                aa, oa, kappa = results['AA'], results['OA'], results['Kappa']
                if aa > best_aa:
                    best_aa, best_oa, best_kappa = aa, oa, kappa
                    torch.save(model.state_dict(), model_path)

            if epoch % args.display_interval == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, OA: {oa:.4f}, AA: {aa:.4f}, Kappa: {kappa:.4f}')

        model.load_state_dict(torch.load(model_path))
        pred = model(data).argmax(dim=1)
        
        results = compute_metrics(
                    prediction=pred.detach().cpu().numpy(), #pred[data.test_mask].detach().cpu().numpy(),
                    target=data.y.detach().cpu().numpy(), # data.y[data.test_mask].detach().cpu().numpy(),
                    n_classes=int(data.y.max()),
                    ignored_labels=[ignored_label]
                )
        statistics.append(results)
        cm_viz(cm=results["Confusion matrix"], labels_text=labels_text, replica=rep)
        metrics2text(results=results, labels_text=labels_text, replica=rep)
    # ----------------------------------------------------------------

    show_statistics(statistics, labels_text)
