import os

from matplotlib import pyplot as plt
import numpy as np
import torch

from model.byol_trainer import BYOLTrainer
from model.gnn import Gnn
from utils.adjacency import meta
from utils.report import compute_metrics, metrics2text, show_statistics
from utils.toolbox import initialize, get_timestamp
from utils.visualize import convert_to_color_


if __name__ == "__main__":
    """
    The purpose of this script is to: 
    1) load the checkpoint.
    2) reproduce the precision reported in the paper.
    """

    # ----------------------------------------------------------------
    # Specify the dataset.
    dataset_name = "Houston" # "MUUFL" / "Trento" / "Houston"
    args, device, labels_text, ignored_label = initialize(dataset_name)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # Load the checkpoint of BYOL (stage one)
    trainer = BYOLTrainer(args, device, labels_text, ignored_label)
    trainer.reload()
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # Preparation for GAT (stage two)
    statistics = []
    timestamp = get_timestamp()
    gen = meta(args, device, ignored_label)
    train_mask, test_mask = gen.mask_generation()
    feat = trainer.inference(args, train_mask+test_mask)['feats']
    data = gen.pyg_data_generation(feat).to(device)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # Load the checkpoint of GAT (stage two)
    # for rep in range(args.num_replicates):
    for rep in range(9, 10):
        print(f"===== replica {rep} =====")
        checkpoint_dir = os.path.join("checkpoint", f"{dataset_name}", "GAT")
        checkpoint_path = os.path.join(checkpoint_dir, f"model{rep}.pth")

        model = Gnn(args, num_features=data.x.shape[1], num_classes=data.y.max().item() + 1).to(device)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

        pred = model(data).argmax(dim=1)

        arr_2d = np.zeros(gen.gt.shape).astype(np.int64)
        for i, (x, y) in enumerate(gen.indices):
            arr_2d[x, y] = pred[i] + 1
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(convert_to_color_(arr_2d, f"{dataset_name}"))
        plt.savefig(f"visualize/{dataset_name}/MSG.png", dpi=600)

        results = compute_metrics(
                    prediction=pred.detach().cpu().numpy(),
                    target=data.y.detach().cpu().numpy(),
                    n_classes=int(data.y.max()),
                    ignored_labels=[ignored_label]
                )
        statistics.append(results)
        metrics2text(results=results, labels_text=labels_text, replica=rep)
    # ----------------------------------------------------------------

    show_statistics(statistics, labels_text)
