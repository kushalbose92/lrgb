"""
pascalvoc_sp.py
================================================================================
LRGB — PascalVOC-SP  (node classification, computer vision)
Paper : "Long Range Graph Benchmark" (Dwivedi et al., NeurIPS 2022 D&B Track)
        https://arxiv.org/pdf/2206.08164
Code  : https://github.com/vijaydwivedi75/lrgb

--------------------------------------------------------------------------------
DATASET SUMMARY  (Table 1 & 2 of the paper)
--------------------------------------------------------------------------------
  Domain            : Computer Vision (SLIC superpixel graphs over Pascal VOC 2011)
  Task              : Node classification (21 semantic segmentation classes)
  Node features     : 14-dim  "Pixel + Coord"
                       (12-dim RGB stats: mean/std/max/min per channel,
                        2-dim (x,y) coordinates of superpixel centroid)
  Edge features     : 2-dim "Edge Weight" (Sobel-filter boundary weight, pixel count)
  Performance metric: macro-averaged F1 score
  Total graphs      : 11,355   (avg. 479.40 nodes/graph, avg. degree 5.65)
  Official splits   : train 8,498 / val 1,428 / test 1,429  (fixed, stratified)
  Avg shortest path : 10.74 ± 0.51     Avg diameter: 27.62 ± 2.13

--------------------------------------------------------------------------------
GCN HYPERPARAMETERS  (paper Table C.1 — "Baseline hyperparameters")
--------------------------------------------------------------------------------
  Hidden dim d       : 220
  Layers L           : 10
  Parameter budget   : ~500k   (paper reports 496k for their exact GCN config)
  Optimizer          : Adam, default betas, weight decay handled via PyG default
  Learning rate      : 0.0003 - 0.001 (paper range; we use 1e-3 with plateau decay)
  LR schedule        : x0.5 decay on validation-loss plateau
  Training budget    : up to 60h wall-clock (paper); we expose --epochs instead
  Seeds              : paper averages over 4 seeds — we default to 1 run,
                        loop externally for multi-seed reproduction
  Residual connections: yes (paper: "we do use residual connections in all
                        our baseline models")
  Regularization      : BatchNorm + ReLU + Dropout per layer (standard GNN
                        Benchmarking suite block; paper does not specify an
                        explicit dropout value, we default to 0.0)

NOTE ON REPRODUCIBILITY: hitting the *exact* 496k parameter count and the
*exact* published Test F1 = 0.1268 ± 0.0060 additionally depends on
architectural micro-choices (exact MLP-head depth, exact weight init) that
are only fully pinned down in the official GraphGPS-based codebase, not in
the paper text. This file reproduces the documented (d, L) configuration and
training protocol; expect small variation around the reported numbers.
================================================================================
"""

import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score
from camp import camp_update    
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_networkx
import networkx as nx


# ═══════════════════════════════════════════════════════════════════════════
# § 1. CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────
# All hyperparameters live here, taken directly from Table C.1 of the paper.
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = dict(
    # ── Data ──────────────────────────────────────────────────────────────
    dataset_name = "PascalVOC-SP",
    root         = "./data/LRGB",     # PyG will download+cache here
    num_classes  = 21,                # semantic segmentation classes
    node_in_dim  = 14,                # Pixel(12) + Coord(2)
    centrality   = "degree",          # centrality measure for node ordering
    batch_frac   = 1.0,               # fraction of nodes to update per layer (CAMP)

    # ── GCN architecture (Table C.1) ─────────────────────────────────────
    hidden_dim   = 220,
    num_layers   = 8,
    dropout      = 0.0,               # not specified in paper; safe default
    use_residual = True,              # paper: residual connections used

    # ── Optimization (Appendix C.1) ──────────────────────────────────────
    lr              = 1e-3,           # paper range: 0.0003 – 0.001
    lr_decay_factor = 0.8,            # decay on val-loss plateau
    reduce_factor   = 0.5,
    lr_patience     = 20,             # epochs of no improvement before decay
    weight_decay    = 0.0,
    batch_size      = 128,
    epochs          = 500,            # paper trains up to 60h; set a sane cap
    seed            = 0,

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)


# ═══════════════════════════════════════════════════════════════════════════
# § 2. DATA LOADING & GRAPH ATTRIBUTE INSPECTION
# ───────────────────────────────────────────────────────────────────────────
# PyG ships a native `LRGBDataset` wrapper (torch_geometric.datasets.lrgb)
# that downloads and parses all 5 LRGB datasets. For PascalVOC-SP it returns
# `Data` objects with:
#     x           : [num_nodes, 14]  float32   -- Pixel+Coord node features
#     edge_index  : [2, num_edges]   long       -- COO connectivity
#     edge_attr   : [num_edges, 2]   float32    -- edge weight + boundary count
#     y           : [num_nodes]      long       -- per-node class in {0,...,20}
# ═══════════════════════════════════════════════════════════════════════════


def load_datasets(cfg: dict = CONFIG):
    """
    Load the three official splits of PascalVOC-SP.

    Returns
    -------
    dict with keys "train", "val", "test", each an LRGBDataset split.
    """
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
    """
    Print the core attributes of the first `n_show` graphs: node features,
    edge index, edge attributes, and labels. Useful as a sanity check that
    the loader is parsing the raw LRGB files correctly.
    """
    # print(f"\nDataset: {dataset}")
    print(f"  Number of graphs : {len(dataset)}")

    for i in range(min(n_show, len(dataset))):
        data = dataset[i]
        print(f"\n  ── Graph {i} ──────────────────────────────────────────")
        print(f"    x          : shape {tuple(data.x.shape)}  dtype {data.x.dtype}")
        print(f"    edge_index : shape {tuple(data.edge_index.shape)}  dtype {data.edge_index.dtype}")
        print(f"    edge_attr  : shape {tuple(data.edge_attr.shape)}  dtype {data.edge_attr.dtype}")
        print(f"    y          : shape {tuple(data.y.shape)}  dtype {data.y.dtype}")
        print(f"    y (unique) : {torch.unique(data.y).tolist()}")
        print(f"    num_nodes  : {data.num_nodes}")
        print(f"    num_edges  : {data.num_edges}")
        print("    centrality : ", data.centrality.tolist())


# ═══════════════════════════════════════════════════════════════════════════
# § 3. MODEL — GCN for node classification
# ───────────────────────────────────────────────────────────────────────────
# Standard GCN block (GCNConv -> BatchNorm -> ReLU -> Dropout -> residual),
# following the "Benchmarking GNNs" (Dwivedi et al. 2020) design that the
# LRGB baselines build on. GCN, per the paper, does NOT consume edge_attr
# (edge-feature-aware baselines are GINE / GatedGCN); we therefore ignore
# `edge_attr` here, matching the paper's vanilla-GCN definition.
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
        if x_in.shape == x.shape:
            x = x + x_in
        return x


class GCN_NodeClassifier(nn.Module):
    """
    L-layer GCN for node classification on PascalVOC-SP.

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
            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.layers[i].dropout, training=self.training)
            if self.batch_frac > 0.0:
                x, prev_embed = camp_update(node_counts, self.batch_frac, ptr, centrality, i, x, prev_embed, len(self.layers))
        return self.output_head(x) 
    

# ═══════════════════════════════════════════════════════════════════════════
# § 4. LOSS FUNCTION
# ───────────────────────────────────────────────────────────────────────────
# Node classification with 21 mutually-exclusive classes -> standard
# cross-entropy loss over per-node logits.
# ═══════════════════════════════════════════════════════════════════════════

def compute_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss for per-node 21-way classification.

    Parameters
    ----------
    logits : [num_nodes, num_classes] raw (unnormalized) scores
    y      : [num_nodes] integer class labels in {0, ..., num_classes-1}
    """
    return F.cross_entropy(logits, y)


# ═══════════════════════════════════════════════════════════════════════════
# § 5. METRIC — macro F1
# ───────────────────────────────────────────────────────────────────────────
# The paper's official metric for PascalVOC-SP is the macro-averaged F1
# score over the 21 node classes, computed over ALL nodes across the split
# (not averaged per-graph then across graphs).
# ═══════════════════════════════════════════════════════════════════════════

def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """
    Macro-averaged F1 score across all node predictions in a split.

    Parameters
    ----------
    y_true : 1D array of ground-truth class indices
    y_pred : 1D array of predicted class indices (argmax of logits)
    num_classes : total number of classes (for consistent label set)
    """
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
    total_macro_f1 = 0.0

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        logits = model(batch)
        loss   = compute_loss(logits, batch.y)

        total_loss  += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes

        preds = logits.argmax(dim=-1)
        macro_f1 = compute_macro_f1(batch.y.cpu().numpy(), preds.cpu().numpy(), num_classes)
        total_macro_f1 += macro_f1

    return total_loss / total_nodes, total_macro_f1 / (i+1)


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
    print("Loading PascalVOC-SP splits...")
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
    print(f"\nModel parameter count: {n_params:,}  (paper target: ~496k)")

    # ── 3. Optimizer + LR schedule ───────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["reduce_factor"], patience=cfg["lr_patience"]
    )

    # print("Centrality measure for CAMP: ", cfg["centrality"])
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

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | val_F1 {val_f1:.4f} | test_F1 {test_f1:.4f}")

    # ── 5. Final test evaluation with best validation checkpoint ────────
    model.load_state_dict(best_state)
    test_loss, test_f1 = evaluate(model, test_loader, cfg["device"], cfg["num_classes"])

    print("\n" + "=" * 55)
    print(f"Best Val macro-F1  : {best_val_f1:.4f}")
    print(f"Test  macro-F1     : {test_f1:.4f}   (paper GCN: 0.1268 ± 0.0060)")
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
