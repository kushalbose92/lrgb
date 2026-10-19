"""
coco_sp.py
================================================================================
LRGB — COCO-SP  (node classification, computer vision)
Paper : "Long Range Graph Benchmark" (Dwivedi et al., NeurIPS 2022 D&B Track)
        https://arxiv.org/pdf/2206.08164
Code  : https://github.com/vijaydwivedi75/lrgb

--------------------------------------------------------------------------------
DATASET SUMMARY  (Table 1 & 2 of the paper)
--------------------------------------------------------------------------------
  Domain            : Computer Vision (SLIC superpixel graphs over MS COCO)
  Task              : Node classification (81 semantic segmentation classes)
  Node features     : 14-dim  "Pixel + Coord"  (same scheme as PascalVOC-SP)
  Edge features     : 2-dim "Edge Weight"
  Performance metric: macro-averaged F1 score
  Total graphs      : 123,286  (avg. 476.88 nodes/graph, avg. degree 5.65)
  Official splits   : train 113,286 / val 5,000 / test 5,000
  Avg shortest path : 10.66 ± 0.55     Avg diameter: 27.39 ± 2.14

  NOTE: COCO-SP is ~11x larger than PascalVOC-SP (123k vs 11k graphs,
  58.8M vs 5.4M total nodes). Downloading and processing this dataset is
  substantially slower and requires up to 72GB RAM per the paper's compute
  notes (Appendix C.2). Budget accordingly.

--------------------------------------------------------------------------------
GCN HYPERPARAMETERS  (paper Table C.1 — "Baseline hyperparameters")
--------------------------------------------------------------------------------
  Hidden dim d       : 220
  Layers L           : 8
  Parameter budget   : ~500k   (paper reports 509k for their exact GCN config)
  Optimizer          : Adam, default betas
  Learning rate      : 0.0003 - 0.001 (paper range; we use 1e-3 with plateau decay)
  LR schedule        : x0.5 decay on validation-loss plateau
  Training budget    : up to 60h wall-clock (paper); avg epoch time ~111s on A100
  Seeds              : paper averages over 4 seeds
  Residual connections: yes
  Regularization      : BatchNorm + ReLU + Dropout per layer; dropout value
                        not specified in paper, default 0.0 here

NOTE ON REPRODUCIBILITY: see pascalvoc_sp.py header for the same caveat on
exact parameter-count / metric matching. This file reproduces the documented
(d, L) configuration and training protocol from Table C.1.
================================================================================
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score
import argparse
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_networkx
from camp import camp_update


# ═══════════════════════════════════════════════════════════════════════════
# § 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = dict(
    # ── Data ──────────────────────────────────────────────────────────────
    dataset_name = "COCO-SP",
    root         = "./data/LRGB",
    num_classes  = 81,                # semantic segmentation classes (COCO)
    node_in_dim  = 14,                # Pixel(12) + Coord(2)
    centrality   = "degree",          # centrality measure for node ordering
    batch_frac = 1.0, 

    # ── GCN architecture (Table C.1) ─────────────────────────────────────
    hidden_dim   = 220,
    num_layers   = 8,
    dropout      = 0.0,
    use_residual = True,

    # ── Optimization (Appendix C.1) ──────────────────────────────────────
    lr              = 1e-3,
    lr_decay_factor = 0.8,
    reduce_factor   = 0.5,
    lr_patience     = 20,
    weight_decay    = 0.0,
    batch_size      = 128,
    epochs          = 500,            # COCO-SP is large; fewer epochs than VOC by default
    seed            = 0,

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)


# ═══════════════════════════════════════════════════════════════════════════
# § 2. DATA LOADING & GRAPH ATTRIBUTE INSPECTION
# ───────────────────────────────────────────────────────────────────────────
# Same tensor layout as PascalVOC-SP:
#     x           : [num_nodes, 14]  float32
#     edge_index  : [2, num_edges]   long
#     edge_attr   : [num_edges, 2]   float32
#     y           : [num_nodes]      long, in {0, ..., 80}  (already remapped
#                   to contiguous indices by PyG's LRGBDataset internally --
#                   see `label_remap_coco` in torch_geometric.datasets.lrgb)
# ═══════════════════════════════════════════════════════════════════════════

def load_datasets(cfg: dict = CONFIG):
    """Load the three official splits of COCO-SP."""
    os.makedirs(cfg["root"], exist_ok=True)

    train_set = LRGBDataset(root=cfg["root"], name=cfg["dataset_name"], split="train")
    val_set   = LRGBDataset(root=cfg["root"], name=cfg["dataset_name"], split="val")
    test_set  = LRGBDataset(root=cfg["root"], name=cfg["dataset_name"], split="test")
    data_splits = [list(train_set), list(val_set), list(test_set)]
    print("Centrality measure for CAMP: ", cfg["centrality"])
    print("Pre-processing begins...")
    data_splits = add_centrality(data_splits, cfg)
    print("Pre-processing done...")
    train_set, val_set, test_set = data_splits[0], data_splits[1], data_splits[2]

    return dict(train=train_set, val=val_set, test=test_set)


def inspect_graph_attributes(dataset, n_show: int = 1):
    """Print core attributes of the first `n_show` graphs (sanity check)."""
    print(f"\nDataset: {dataset}")
    print(f"  Number of graphs : {len(dataset)}")

    for i in range(min(n_show, len(dataset))):
        data = dataset[i]
        print(f"\n  ── Graph {i} ──────────────────────────────────────────")
        print(f"    x          : shape {tuple(data.x.shape)}  dtype {data.x.dtype}")
        print(f"    edge_index : shape {tuple(data.edge_index.shape)}  dtype {data.edge_index.dtype}")
        print(f"    edge_attr  : shape {tuple(data.edge_attr.shape)}  dtype {data.edge_attr.dtype}")
        print(f"    y          : shape {tuple(data.y.shape)}  dtype {data.y.dtype}")
        print(f"    y (unique) : {torch.unique(data.y).tolist()[:10]} ...")
        print(f"    num_nodes  : {data.num_nodes}")
        print(f"    num_edges  : {data.num_edges}")
        print("    centrality : ", data.centrality.tolist())


# ═══════════════════════════════════════════════════════════════════════════
# § 3. MODEL — GCN for node classification
# ───────────────────────────────────────────────────────────────────────────
# Identical block design to PascalVOC-SP: vanilla GCN ignores edge_attr,
# matching the paper's "GCN" baseline definition.
# ═══════════════════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    """One GCN message-passing block with BatchNorm, ReLU, Dropout, residual."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.conv    = GCNConv(in_dim, out_dim)
        self.bn      = nn.BatchNorm1d(out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_in = x
        x = self.conv(x, edge_index)
        # x = self.bn(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        if x_in.shape == x.shape:
            x = x + x_in
        return x


class GCN_NodeClassifier(nn.Module):
    """
    L-layer GCN for node classification on COCO-SP.

    Architecture
    ------------
    Linear input encoder (14 -> hidden_dim)
      -> L x GCNLayer (hidden_dim -> hidden_dim)
      -> Linear output head (hidden_dim -> num_classes), applied per node.
    """

    def __init__(
        self,
        in_dim      : int = CONFIG["node_in_dim"],
        hidden_dim  : int = CONFIG["hidden_dim"],
        num_layers  : int = CONFIG["num_layers"],
        num_classes : int = CONFIG["num_classes"],
        dropout     : float = CONFIG["dropout"],
        batch_frac  : float = CONFIG["batch_frac"]
    ):
        super().__init__()
        self.input_encoder = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GCNLayer(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.output_head = nn.Linear(hidden_dim, num_classes)
        self.batch_frac = batch_frac

    def forward(self, batch) -> torch.Tensor:
        x = batch.x
        edge_index = batch.edge_index
        if self.batch_frac > 0.0:
            centrality = batch.centrality
        x_batch = batch.batch
        ptr = batch.ptr
        x = self.input_encoder(x)
        prev_embed = x
        _, node_counts = torch.unique(x_batch, return_counts = True)
        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index)
            if 1 != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.layers[i].dropout, training=self.training)
            if self.batch_frac > 0.0:
                x, prev_embed = camp_update(node_counts, self.batch_frac, ptr, centrality, i, x, prev_embed, len(self.layers))
        return self.output_head(x)


# ═══════════════════════════════════════════════════════════════════════════
# § 4. LOSS FUNCTION
# ───────────────────────────────────────────────────────────────────────────
# Same as PascalVOC-SP: 81-way per-node classification -> cross-entropy.
# ═══════════════════════════════════════════════════════════════════════════

def compute_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for per-node 81-way classification."""
    return F.cross_entropy(logits, y)


# ═══════════════════════════════════════════════════════════════════════════
# § 5. METRIC — macro F1
# ═══════════════════════════════════════════════════════════════════════════

def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Macro-averaged F1 score across all node predictions in a split."""
    labels = list(range(num_classes))
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


# ═══════════════════════════════════════════════════════════════════════════
# § 6. TRAIN / EVAL LOOPS
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device):
    """Run one training epoch; returns average training loss."""
    model.train()
    total_loss, total_nodes = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch)
        loss   = compute_loss(logits, batch.y)

        loss.backward()
        optimizer.step()

        total_loss  += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes

    return total_loss / total_nodes


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """Run evaluation; returns (avg_loss, macro_f1)."""
    model.eval()
    total_loss, total_nodes = 0.0, 0
    # all_true, all_pred = [], []
    total_macro_f1 = 0.0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss   = compute_loss(logits, batch.y)

        total_loss  += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes

        preds = logits.argmax(dim=-1)
        macro_f1 = compute_macro_f1(batch.y.cpu().numpy(), preds.cpu().numpy(), num_classes)
        total_macro_f1 += macro_f1
        # all_true.append(batch.y.cpu().numpy())
        # all_pred.append(preds.cpu().numpy())

    # y_true = np.concatenate(all_true)
    # y_pred = np.concatenate(all_pred)
    # macro_f1 = compute_macro_f1(y_true, y_pred, num_classes)

    return total_loss / total_nodes, total_macro_f1 / len(loader)


# ═══════════════════════════════════════════════════════════════════════════
# § 7. MAIN — end-to-end training to reproduce the GCN baseline
# ═══════════════════════════════════════════════════════════════════════════


# choice of centrality
def centrality_measure(edge_index, num_nodes, centrality):
    pyg_g = Data(edge_index = edge_index, num_nodes = num_nodes)
    nx_G = to_networkx(pyg_g)
    if centrality == 'degree':
        node_degrees = degree(edge_index[0], num_nodes = num_nodes)
        sorted_node_degree, sorted_indices = torch.sort(node_degrees, descending = True)
        return sorted_indices
    elif centrality == 'betweenness':
        centrality = nx.betweenness_centrality(nx_G)
        centrality = torch.tensor(list(centrality.values()))
        centrality = torch.argsort(centrality, descending = True)
        return centrality
    elif centrality == 'pagerank':
        centrality = nx.pagerank(nx_G)
        centrality = torch.tensor(list(centrality.values()))
        centrality = torch.argsort(centrality, descending = True)
        return centrality
    elif centrality == 'load':
        centrality = nx.load_centrality(nx_G)
        centrality = torch.tensor(list(centrality.values()))
        centrality = torch.argsort(centrality, descending = True)
        return centrality
    elif centrality == 'closeness':
        centrality = nx.closeness_centrality(nx_G)
        centrality = torch.tensor(list(centrality.values()))
        centrality = torch.argsort(centrality, descending = True)
        return centrality
    else:
        raise ValueError(f"Incorrect centrality: {centrality}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train PascalVOC-SP with a selectable CAMP centrality measure.")
    parser.add_argument(
        "--centrality",
        type=str,
        choices=["degree", "betweenness", "pagerank", "load", "closeness"],
        default=CONFIG["centrality"],
        help="Centrality measure used for node ordering in the CAMP stage.",
    )
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--root", type=str, default=CONFIG["root"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    return parser.parse_args()


def add_centrality(data_splits, cfg):
    for split in data_splits:
        for graph in split:
            graph.centrality = centrality_measure(graph.edge_index, graph.num_nodes, centrality=cfg['centrality'])
    return data_splits


def main(cfg: dict = CONFIG):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("Loading COCO-SP splits... (this is a large dataset — 123k graphs)")
    datasets = load_datasets(cfg)
    inspect_graph_attributes(datasets["train"], n_show=1)

    train_loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(datasets["val"],   batch_size=cfg["batch_size"], shuffle=False)
    test_loader  = DataLoader(datasets["test"],  batch_size=cfg["batch_size"], shuffle=False)

    # ── 2. Build model ───────────────────────────────────────────────────
    model = GCN_NodeClassifier(
        in_dim      = cfg["node_in_dim"],
        hidden_dim  = cfg["hidden_dim"],
        num_layers  = cfg["num_layers"],
        num_classes = cfg["num_classes"],
        dropout     = cfg["dropout"],
    ).to(cfg["device"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameter count: {n_params:,}  (paper target: ~509k)")

    # ── 3. Optimizer + LR schedule ───────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["reduce_factor"], patience=cfg["lr_patience"]
    )

    # ── 4. Training loop with best-checkpoint selection on val macro-F1 ──
    best_val_f1 = -1.0
    best_state  = None

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg["device"])
        train_loss, train_f1 = evaluate(model, train_loader, cfg["device"], cfg["num_classes"])
        val_loss, val_f1 = evaluate(model, val_loader, cfg["device"], cfg["num_classes"])
        test_loss, test_f1 = evaluate(model, test_loader, cfg["device"], cfg["num_classes"])
        scheduler.step(train_loss)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | val_F1 {val_f1:.4f} | test_F1 {test_f1:.4f}")

    # ── 5. Final test evaluation with best validation checkpoint ────────
    model.load_state_dict(best_state)
    test_loss, test_f1 = evaluate(model, test_loader, cfg["device"], cfg["num_classes"])

    print("\n" + "=" * 55)
    print(f"Best Val macro-F1  : {best_val_f1:.4f}")
    print(f"Test  macro-F1     : {test_f1:.4f}   (paper GCN: 0.0841 ± 0.0010)")
    print("=" * 55)

    return dict(model=model, test_f1=test_f1, best_val_f1=best_val_f1)


if __name__ == "__main__":
    args = parse_args()
    cfg = CONFIG.copy()
    cfg.update({
        "centrality": args.centrality,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "root": args.root,
        "lr": args.lr,
    })
    main(cfg)
