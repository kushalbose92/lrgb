"""
peptides_struct.py
================================================================================
LRGB — Peptides-struct  (multi-target graph regression, chemistry)
Paper : "Long Range Graph Benchmark" (Dwivedi et al., NeurIPS 2022 D&B Track)
        https://arxiv.org/pdf/2206.08164
Code  : https://github.com/vijaydwivedi75/lrgb

--------------------------------------------------------------------------------
DATASET SUMMARY  (Table 1 & 2, Section 3.4.2, Appendix A.2 of the paper)
--------------------------------------------------------------------------------
  Domain            : Chemistry (SAME peptide molecular graphs as
                       Peptides-func, different task)
  Task              : Multi-target graph regression -- predict 11 aggregated
                       3D structural properties of each peptide, each
                       normalized to zero mean / unit standard deviation:
                         Inertia_mass (x3 principal components)
                         Inertia_valence (x3 principal components)
                         Length (x3 principal axes)
                         Sphericity
                         Plane_best_fit
                       (11 real-valued targets total; see Appendix A.2)
  Node features     : 9-dim  OGB "Atom Encoder" categorical codes
  Edge features     : 3-dim  OGB "Bond Encoder" categorical codes
  Performance metric: Mean Absolute Error (MAE), averaged over the 11
                       targets. Coefficient of Determination (R2) is also
                       tracked as an auxiliary metric (Appendix A.3).
  Total graphs      : 15,535  (identical graphs to Peptides-func)
  Official splits   : identical to Peptides-func (structure correlates with
                       function, so the same split preserves distribution)

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

  Paper GCN results  : Train MAE 0.2939 ± 0.0055 | Test MAE 0.3496 ± 0.0013
                       Train R2  0.6513 ± 0.0078 | Test R2  0.6019 ± 0.0027

NOTE (from a later re-assessment, Tönshoff et al. 2023, "Where Did the Gap
Go?"): GCN's test MAE improves substantially with a deeper (2-3 layer) MLP
prediction head instead of the single linear layer used in the original
paper. This file uses the original paper's linear head for a faithful
reproduction of Table 4/A.3; swap `output_head` for an MLP if you want to
chase the improved numbers from the reassessment paper instead.
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
from sklearn.metrics import r2_score


# ═══════════════════════════════════════════════════════════════════════════
# § 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = dict(
    # ── Data ──────────────────────────────────────────────────────────────
    dataset_name = "Peptides-struct",
    root         = "./data/LRGB",
    num_targets  = 11,                # 11 real-valued structural properties

    # ── GCN architecture (Table C.1) ─────────────────────────────────────
    hidden_dim   = 300,
    num_layers   = 5,
    dropout      = 0.0,
    use_residual = True,

    # ── Optimization (Appendix C.1) ──────────────────────────────────────
    lr              = 1e-3,
    lr_decay_factor = 0.8,
    lr_patience     = 20,
    weight_decay    = 0.0,
    reduce_factor   = 0.5,
    batch_size      = 128,
    epochs          = 500,
    seed            = 0,

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)


# ═══════════════════════════════════════════════════════════════════════════
# § 2. DATA LOADING & GRAPH ATTRIBUTE INSPECTION
# ───────────────────────────────────────────────────────────────────────────
#     x           : [num_nodes, 9]   long    -- OGB atom features
#     edge_index  : [2, num_edges]   long
#     edge_attr   : [num_edges, 3]   long    -- OGB bond features
#     y           : [1, 11]          float   -- 11 regression targets,
#                    z-score normalized, one row per GRAPH
# ═══════════════════════════════════════════════════════════════════════════

def load_datasets(cfg: dict = CONFIG):
    """Load the three official splits of Peptides-struct."""
    os.makedirs(cfg["root"], exist_ok=True)

    train_set = LRGBDataset(root=cfg["root"], name=cfg["dataset_name"], split="train")
    val_set   = LRGBDataset(root=cfg["root"], name=cfg["dataset_name"], split="val")
    test_set  = LRGBDataset(root=cfg["root"], name=cfg["dataset_name"], split="test")

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
# § 3. MODEL — GCN for graph-level multi-target regression
# ───────────────────────────────────────────────────────────────────────────
# Same encoder/backbone as Peptides-func (same underlying graphs), but the
# output head regresses 11 real-valued, z-score-normalized targets instead
# of 10 binary labels.
# ═══════════════════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    """One GCN message-passing block with BatchNorm, ReLU, Dropout, residual."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.conv    = GCNConv(in_dim, out_dim)
        self.bn      = nn.BatchNorm1d(out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, edge_index)
        return x


class GCN_GraphRegressor(nn.Module):
    """
    L-layer GCN for multi-target graph regression on Peptides-struct.

    Architecture
    ------------
    AtomEncoder (9-dim OGB codes -> hidden_dim)
      -> L x GCNLayer (hidden_dim -> hidden_dim)
      -> global_mean_pool over nodes in each graph -> [batch, hidden_dim]
      -> Linear output head (hidden_dim -> num_targets), raw regression output
    """

    def __init__(
        self,
        hidden_dim  : int = CONFIG["hidden_dim"],
        num_layers  : int = CONFIG["num_layers"],
        num_targets : int = CONFIG["num_targets"],
        dropout     : float = CONFIG["dropout"],
    ):
        super().__init__()
        self.input_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.layers = nn.ModuleList(
            [GCNLayer(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.output_head = nn.Linear(hidden_dim, num_targets)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.input_encoder(x)
        for i in range(len(self.layers)):
            h = self.layers[i](h, edge_index)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.layers[i].dropout, training=self.training)
        h = self.output_head(h)
        h_graph = global_mean_pool(h, batch)         
        return h_graph   


# ═══════════════════════════════════════════════════════════════════════════
# § 4. LOSS FUNCTION
# ───────────────────────────────────────────────────────────────────────────
# Regression on 11 z-score-normalized targets -> L1 (Mean Absolute Error)
# loss, matching the paper's chosen training/eval metric for this dataset
# ("we use the Mean Absolute Error (MAE), typically selected for molecular
# property regression datasets").
# ═══════════════════════════════════════════════════════════════════════════

def compute_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error (L1) loss for multi-target regression.

    Parameters
    ----------
    pred : [num_graphs, num_targets] predicted values
    y    : [num_graphs, num_targets] ground-truth (z-score normalized) values
    """
    return F.l1_loss(pred, y)


# ═══════════════════════════════════════════════════════════════════════════
# § 5. METRICS — MAE (primary) and R2 (auxiliary, per Appendix A.3)
# ═══════════════════════════════════════════════════════════════════════════

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error, averaged across all 11 targets and all graphs."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mean_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Unweighted mean Coefficient of Determination (R2) across the 11 targets,
    matching the paper's Appendix A.3 "extended evaluation metrics" table.
    """
    r2_list = [
        r2_score(y_true[:, t], y_pred[:, t]) for t in range(y_true.shape[1])
    ]
    return float(np.mean(r2_list))


# ═══════════════════════════════════════════════════════════════════════════
# § 6. TRAIN / EVAL LOOPS
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device):
    """Run one training epoch; returns average training loss (MAE)."""
    model.train()
    total_loss, total_graphs = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch.x, batch.edge_index, batch.batch)
        y    = batch.y.view(pred.shape).float()
        loss = compute_loss(pred, y)

        loss.backward()
        optimizer.step()

        n_graphs = batch.num_graphs
        total_loss   += loss.item() * n_graphs
        total_graphs += n_graphs

    return total_loss / total_graphs


@torch.no_grad()
def evaluate(model, loader, device):
    """Run evaluation; returns (mae, r2)."""
    model.eval()
    total_loss, total_graphs = 0.0, 0
    all_true, all_pred = [], []
    total_mae = 0.0

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch)
        y    = batch.y.view(pred.shape).float()
        loss = compute_loss(pred, y)

        n_graphs = batch.num_graphs
        total_loss += loss.item() * n_graphs
        total_graphs += n_graphs
        
        mae = compute_mae(y.cpu().numpy(), pred.cpu().numpy())
        total_mae += mae

    mae = total_mae / (i+1)
    # r2  = compute_mean_r2(y, pred)

    return mae, total_loss/total_graphs


# ═══════════════════════════════════════════════════════════════════════════
# § 7. MAIN — end-to-end training to reproduce the GCN baseline
# ═══════════════════════════════════════════════════════════════════════════

def main(cfg: dict = CONFIG):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("Loading Peptides-struct splits...")
    datasets = load_datasets(cfg)
    inspect_graph_attributes(datasets["train"], n_show=1)

    train_loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(datasets["val"],   batch_size=cfg["batch_size"], shuffle=False)
    test_loader  = DataLoader(datasets["test"],  batch_size=cfg["batch_size"], shuffle=False)

    # ── 2. Build model ───────────────────────────────────────────────────
    model = GCN_GraphRegressor(
        hidden_dim  = cfg["hidden_dim"],
        num_layers  = cfg["num_layers"],
        num_targets = cfg["num_targets"],
        dropout     = cfg["dropout"],
    ).to(cfg["device"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameter count: {n_params:,}  (paper target: ~508k)")

    # ── 3. Optimizer + LR schedule ───────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["reduce_factor"], patience=cfg["lr_patience"]
    )

    # ── 4. Training loop with best-checkpoint selection on val MAE ──────
    # Lower MAE is better, so we track the minimum.
    best_val_mae = float("inf")
    best_state   = None

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg["device"])
        train_mae, train_loss = evaluate(model, train_loader, cfg["device"])
        val_mae, val_loss = evaluate(model, val_loader, cfg["device"])
        test_mae, test_loss = evaluate(model, test_loader, cfg["device"])
        scheduler.step(train_loss)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state   = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_MAE {train_mae:.4f} | "
                  f"val_MAE {val_mae:.4f} | test_MAE {test_mae:.4f}")

    # ── 5. Final test evaluation with best validation checkpoint ────────
    model.load_state_dict(best_state)
    test_mae, test_loss = evaluate(model, test_loader, cfg["device"])

    print("\n" + "=" * 55)
    print(f"Best Val MAE : {best_val_mae:.4f}")
    print(f"Test  MAE    : {test_mae:.4f}   (paper GCN: 0.3496 ± 0.0013)")
    # print(f"Test  R2     : {test_r2:.4f}   (paper GCN: 0.6019 ± 0.0027)")
    print("=" * 55)

    return dict(model=model, test_mae=test_mae, best_val_mae=best_val_mae)


if __name__ == "__main__":
    main()
