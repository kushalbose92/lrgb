"""
peptides_func.py
================================================================================
LRGB — Peptides-func  (multi-label graph classification, chemistry)
Paper : "Long Range Graph Benchmark" (Dwivedi et al., NeurIPS 2022 D&B Track)
        https://arxiv.org/pdf/2206.08164
Code  : https://github.com/vijaydwivedi75/lrgb

--------------------------------------------------------------------------------
DATASET SUMMARY  (Table 1 & 2, Section 3.4.1 of the paper)
--------------------------------------------------------------------------------
  Domain            : Chemistry (peptide molecular graphs, from SATPdb)
  Task              : Multi-label graph classification -- predict which of
                       10 peptide function classes apply (Antibacterial,
                       Antiviral, cell-cell communication, etc.). A peptide
                       can belong to several classes at once (avg. 1.65/10).
                       Labels are imbalanced: 16.5% positive overall, ranging
                       from 62.7% (richest class) to 1.9% (poorest class).
  Node features     : 9-dim  OGB "Atom Encoder" categorical codes
                       (nodes = heavy, non-hydrogen atoms)
  Edge features     : 3-dim  OGB "Bond Encoder" categorical codes
  Performance metric: Unweighted mean Average Precision (AP), i.e. area
                       under the precision-recall curve, averaged over the
                       10 label columns (same metric family as ogbg-molpcba)
  Total graphs      : 15,535  (avg. 150.94 nodes/graph, avg. degree 2.04 --
                       much larger than typical molecular benchmarks like
                       ZINC (avg. 23-26 nodes), by design, to require LRI)
  Official splits   : stratified 70% / 15% / 15% (train / val / test) via
                       meta-class stratification on the label combination

--------------------------------------------------------------------------------
GCN HYPERPARAMETERS  (paper Table C.1 — "Baseline hyperparameters")
--------------------------------------------------------------------------------
  Hidden dim d       : 300
  Layers L           : 5
  Parameter budget   : ~500k   (paper reports 508k for their exact GCN config)
  Optimizer          : Adam, default betas
  Learning rate      : 0.0003 - 0.001 (paper range; we use 1e-3 with plateau decay)
  LR schedule        : x0.5 decay on validation-loss plateau
  Seeds              : paper averages over 4 seeds
  Residual connections: yes

  Paper GCN results  : Train AP 0.8840 ± 0.0131 | Test AP 0.5930 ± 0.0023
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
from torch_geometric.nn import GCNConv, global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder
from sklearn.metrics import average_precision_score


# ═══════════════════════════════════════════════════════════════════════════
# § 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = dict(
    # ── Data ──────────────────────────────────────────────────────────────
    dataset_name = "Peptides-func",
    root         = "./data/LRGB",
    num_classes  = 10,                # 10 peptide function labels (multi-label)

    # ── GCN architecture (Table C.1) ─────────────────────────────────────
    hidden_dim   = 300,
    num_layers   = 5,
    dropout      = 0.0,
    use_residual = True,

    # ── Optimization (Appendix C.1) ──────────────────────────────────────
    lr              = 0.001,
    min_lr          = 1e-5,
    lr_decay_factor = 0.8,
    lr_patience     = 20,
    weight_decay    = 1e-5,
    reduce_factor   = 0.5,
    batch_size      = 128,
    epochs          = 500,
    seed            = 42,

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)


# ═══════════════════════════════════════════════════════════════════════════
# § 2. DATA LOADING & GRAPH ATTRIBUTE INSPECTION
# ───────────────────────────────────────────────────────────────────────────
#     x           : [num_nodes, 9]   long    -- OGB atom features
#     edge_index  : [2, num_edges]   long
#     edge_attr   : [num_edges, 3]   long    -- OGB bond features
#     y           : [1, 10]          float   -- multi-label targets (0/1),
#                    one row per GRAPH (graph-level label, not per-node)
# ═══════════════════════════════════════════════════════════════════════════

def load_datasets(cfg: dict = CONFIG):
    """Load the three official splits of Peptides-func."""
    os.makedirs(cfg["root"], exist_ok=True)

    train_set = LRGBDataset(root=cfg["root"], name=cfg["dataset_name"], split="train")
    val_set   = LRGBDataset(root=cfg["root"], name=cfg["dataset_name"], split="val")
    test_set  = LRGBDataset(root=cfg["root"], name=cfg["dataset_name"], split="test")
    print(len(train_set), len(val_set), len(test_set))

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
        print(f"    y          : shape {tuple(data.y.shape)}  dtype {data.y.dtype}  values {data.y}")
        print(f"    num_nodes  : {data.num_nodes}")
        print(f"    num_edges  : {data.num_edges}")


# ═══════════════════════════════════════════════════════════════════════════
# § 3. MODEL — GCN for graph-level multi-label classification
# ───────────────────────────────────────────────────────────────────────────
# AtomEncoder input layer (9-dim OGB codes -> hidden_dim), L GCN layers,
# global mean pooling to a single graph embedding, then a linear multi-label
# classification head. edge_attr is ignored (vanilla GCN baseline).
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
        # if x_in.shape == x.shape:
        #     x = x + x_in
        return x


class GCN_GraphClassifier(nn.Module):
    """
    L-layer GCN for multi-label graph classification on Peptides-func.

    Architecture
    ------------
    AtomEncoder (9-dim OGB codes -> hidden_dim)
      -> L x GCNLayer (hidden_dim -> hidden_dim)
      -> global_mean_pool over nodes in each graph -> [batch, hidden_dim]
      -> Linear output head (hidden_dim -> num_classes)
    """

    def __init__(
        self,
        hidden_dim  : int = CONFIG["hidden_dim"],
        num_layers  : int = CONFIG["num_layers"],
        num_classes : int = CONFIG["num_classes"],
        dropout     : float = CONFIG["dropout"],
    ):
        super().__init__()
        self.input_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.layers = nn.ModuleList(
            [GCNLayer(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.output_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.input_encoder(x)
        for i in range(len(self.layers)):
            h = self.layers[i](h, edge_index)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.layers[i].dropout, training=self.training)
        y_pred  = self.output_head(h)  
        h_graph = global_mean_pool(y_pred, batch)        
        return h_graph          


# ═══════════════════════════════════════════════════════════════════════════
# § 4. LOSS FUNCTION
# ───────────────────────────────────────────────────────────────────────────
# Multi-label classification (each of 10 labels is an independent binary
# decision) -> binary cross-entropy with logits, applied per label then
# averaged. This matches the paper's treatment ("we treat it as a
# multi-label classification as a peptide can belong to several classes
# simultaneously").
# ═══════════════════════════════════════════════════════════════════════════

def compute_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss for multi-label graph classification.

    Parameters
    ----------
    logits : [num_graphs, num_classes] raw scores
    y      : [num_graphs, num_classes] float labels in {0.0, 1.0}
    """
    return F.binary_cross_entropy_with_logits(logits, y)


# ═══════════════════════════════════════════════════════════════════════════
# § 5. METRIC — unweighted mean Average Precision (AP)
# ═══════════════════════════════════════════════════════════════════════════

def compute_mean_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Unweighted mean Average Precision across the 10 label columns.

    Follows the same convention as ogbg-molpcba: AP is computed per label
    column (ignoring columns with no positive examples in the split, which
    would make AP undefined), then averaged uniformly across columns.

    Parameters
    ----------
    y_true  : [num_graphs, num_classes] binary ground-truth array
    y_score : [num_graphs, num_classes] predicted probabilities/scores
    """

    ap_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i], y_score[is_labeled, i])
            ap_list.append(ap)
    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')
    return  sum(ap_list) / len(ap_list)

# ═══════════════════════════════════════════════════════════════════════════
# § 6. TRAIN / EVAL LOOPS
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device):
    """Run one training epoch; returns average training loss."""
    model.train()
    total_loss, total_graphs = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index, batch.batch)
        y      = batch.y.view(logits.shape).float()
        loss   = compute_loss(logits, y)

        loss.backward()
        optimizer.step()

        n_graphs = batch.num_graphs
        total_loss   += loss.item() * n_graphs
        total_graphs += n_graphs

    return total_loss / total_graphs


@torch.no_grad()
def evaluate(model, loader, device):
    """Run evaluation; returns (avg_loss, mean_AP)."""
    model.eval()
    total_loss, total_graphs = 0.0, 0
    all_true, all_score = [], []
    total_mean_ap = 0.0

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        y      = batch.y.view(logits.shape).float()
        loss   = compute_loss(logits, y)

        n_graphs = batch.num_graphs
        total_loss   += loss.item() * n_graphs
        total_graphs += n_graphs

        mean_ap = compute_mean_ap(y.cpu().numpy(), logits.cpu().numpy())
        total_mean_ap += mean_ap 

    return total_loss / total_graphs, total_mean_ap/(i+1)


# ═══════════════════════════════════════════════════════════════════════════
# § 7. MAIN — end-to-end training to reproduce the GCN baseline
# ═══════════════════════════════════════════════════════════════════════════

def main(cfg: dict = CONFIG):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("Loading Peptides-func splits...")
    datasets = load_datasets(cfg)
    inspect_graph_attributes(datasets["train"], n_show=1)

    train_loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(datasets["val"],   batch_size=cfg["batch_size"], shuffle=False)
    test_loader  = DataLoader(datasets["test"],  batch_size=cfg["batch_size"], shuffle=False)

    # ── 2. Build model ───────────────────────────────────────────────────
    model = GCN_GraphClassifier(
        hidden_dim  = cfg["hidden_dim"],
        num_layers  = cfg["num_layers"],
        num_classes = cfg["num_classes"],
        dropout     = cfg["dropout"],
    ).to(cfg["device"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameter count: {n_params:,}  (paper target: ~508k)")

    # ── 3. Optimizer + LR schedule ───────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["reduce_factor"], patience=cfg["lr_patience"]
    )

    # ── 4. Training loop with best-checkpoint selection on val AP ───────
    best_val_ap = -1.0
    best_state  = None

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg["device"])
        train_loss, train_ap = evaluate(model, train_loader, cfg["device"])
        val_loss, val_ap = evaluate(model, val_loader, cfg["device"])
        test_loss, test_ap = evaluate(model, test_loader, cfg["device"])
        scheduler.step(train_loss)

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state  = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | val_AP {val_ap:.4f} | test_AP {test_ap:.4f}")

    # ── 5. Final test evaluation with best validation checkpoint ────────
    model.load_state_dict(best_state)
    test_loss, test_ap = evaluate(model, test_loader, cfg["device"])

    print("\n" + "=" * 55)
    print(f"Best Val AP  : {best_val_ap:.4f}")
    print(f"Test  AP     : {test_ap:.4f}   (paper GCN: 0.5930 ± 0.0023)")
    print("=" * 55)

    return dict(model=model, test_ap=test_ap, best_val_ap=best_val_ap)


if __name__ == "__main__":
    main()
