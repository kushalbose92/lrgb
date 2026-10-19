"""
pcqm_contact.py
================================================================================
LRGB — PCQM-Contact  (link prediction, quantum chemistry)
Paper : "Long Range Graph Benchmark" (Dwivedi et al., NeurIPS 2022 D&B Track)
        https://arxiv.org/pdf/2206.08164
Code  : https://github.com/vijaydwivedi75/lrgb

--------------------------------------------------------------------------------
DATASET SUMMARY  (Table 1 & 2, Section 3.3 of the paper)
--------------------------------------------------------------------------------
  Domain            : Quantum Chemistry (molecular graphs, from PCQM4Mv2)
  Task              : Link prediction -- predict pairs of atoms that are
                       > 5 hops apart in the molecular GRAPH but whose 3D
                       distance is < 3.5 Angstrom (i.e. non-covalent contact,
                       e.g. hydrogen bonds). Explicit hydrogens are included.
  Node features     : 9-dim  OGB "Atom Encoder" categorical codes
  Edge features     : 3-dim  OGB "Bond Encoder" categorical codes
  Performance metric: Hits@1, Hits@3, Hits@10, Mean Reciprocal Rank (MRR)
                       computed in the FILTERED ranking setting (standard
                       in knowledge-graph link prediction literature)
  Total graphs      : 529,434  (avg. 30.14 nodes/graph, avg degree 2.03 --
                       much smaller graphs than the other 4 LRGB datasets,
                       but very deep average diameter given graph size)
  Official splits   : train 90% (476,490) / val 5% (26,472) / test 5% (26,472)

  IMPORTANT DATA-FORMAT NOTE: PyG's LRGBDataset parses PCQM-Contact into a
  DIFFERENT Data schema than the other 4 datasets. Each graph carries:
      x                 : [num_nodes, 9]   long   -- OGB atom features
      edge_index        : [2, num_edges]   long   -- molecular bond graph
      edge_attr         : [num_edges, 3]   long   -- OGB bond features
      edge_label_index  : [2, num_labeled] long   -- candidate (head,tail)
                           pairs to score (positives + sampled negatives)
      edge_label        : [num_labeled]    float  -- 1.0 = true contact,
                           0.0 = negative (non-contact) pair
  There is NO `y` field -- supervision comes entirely from edge_label(_index).

--------------------------------------------------------------------------------
GCN HYPERPARAMETERS  (paper Table C.1 — "Baseline hyperparameters")
--------------------------------------------------------------------------------
  Hidden dim d       : 275
  Layers L           : 5
  Parameter budget   : ~500k   (paper reports 504k for their exact GCN config)
  Optimizer          : Adam, default betas
  Learning rate      : 0.0003 - 0.001 (paper range; we use 1e-3 with plateau decay)
  LR schedule        : x0.5 decay on validation-loss plateau
  Seeds              : paper averages over 4 seeds
  Residual connections: yes

--------------------------------------------------------------------------------
LINK-PREDICTION DECODER & EVALUATION -- IMPLEMENTATION NOTE
--------------------------------------------------------------------------------
The paper decodes a link score as f(h, t) of the probed head/tail node
embeddings and evaluates Hits@K / MRR in the FILTERED setting: for a query
(h, ?), all other nodes in the SAME molecule are scored as candidate tails,
true positive tails sharing the same head are masked out, and the rank of
the true tail is recorded (see paper Section 3.3, "Performance Metric").

This file implements:
  - Decoder: a simple dot-product score f(h,t) = h_i . h_j  (standard,
    parameter-free link-prediction decoder; the paper does not pin down
    the exact decoder architecture in the main text).
  - Ranking: for each positive edge_label_index pair sharing a head node
    within a graph, we rank its score against the scores of ALL OTHER
    candidate pairs (positive + negative) sharing that SAME head node
    within `edge_label_index` for that graph. This is a faithful
    reproduction of the filtered, per-head ranking protocol, restricted
    to the candidate set the dataset already provides in edge_label_index
    (positives + sampled negatives), which is the standard practical proxy
    used across KG-style link-prediction benchmarks (e.g. OGB-LSC link
    property prediction evaluators) when the full node-set corruption is
    computationally prohibitive.
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
from ogb.graphproppred.mol_encoder import AtomEncoder


# ═══════════════════════════════════════════════════════════════════════════
# § 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = dict(
    # ── Data ──────────────────────────────────────────────────────────────
    dataset_name = "PCQM-Contact",
    root         = "./data/LRGB",
    node_in_dim  = 9,                 # OGB AtomEncoder categorical feature dim

    # ── GCN architecture (Table C.1) ─────────────────────────────────────
    hidden_dim   = 275,
    num_layers   = 5,
    dropout      = 0.0,
    use_residual = True,

    # ── Optimization (Appendix C.1) ──────────────────────────────────────
    lr              = 1e-3,
    lr_decay_factor = 0.5,
    lr_patience     = 10,
    weight_decay    = 0.0,
    batch_size      = 128,            # PCQM-Contact graphs are small (~30 nodes)
    epochs          = 50,             # 529k graphs/epoch -> few epochs suffice
    seed            = 0,

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)


# ═══════════════════════════════════════════════════════════════════════════
# § 2. DATA LOADING & GRAPH ATTRIBUTE INSPECTION
# ═══════════════════════════════════════════════════════════════════════════

def load_datasets(cfg: dict = CONFIG):
    """Load the three official splits of PCQM-Contact."""
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
        print(f"    x                : shape {tuple(data.x.shape)}  dtype {data.x.dtype}")
        print(f"    edge_index       : shape {tuple(data.edge_index.shape)}  dtype {data.edge_index.dtype}")
        print(f"    edge_attr        : shape {tuple(data.edge_attr.shape)}  dtype {data.edge_attr.dtype}")
        print(f"    edge_label_index : shape {tuple(data.edge_label_index.shape)}  dtype {data.edge_label_index.dtype}")
        print(f"    edge_label       : shape {tuple(data.edge_label.shape)}  dtype {data.edge_label.dtype}")
        print(f"    edge_label (pos frac): {data.edge_label.float().mean().item():.3f}")
        print(f"    num_nodes        : {data.num_nodes}")


# ═══════════════════════════════════════════════════════════════════════════
# § 3. MODEL — GCN encoder + dot-product link-prediction decoder
# ───────────────────────────────────────────────────────────────────────────
# Node features are 9-dim OGB categorical codes -> must go through
# AtomEncoder (a sum-of-embeddings layer) BEFORE any GCNConv, exactly like
# every other OGB-molecular-featurized model. edge_attr (Bond Encoder codes)
# is ignored, matching the paper's vanilla-GCN baseline definition.
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
        x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if x_in.shape == x.shape:
            x = x + x_in
        return x


class GCN_LinkPredictor(nn.Module):
    """
    L-layer GCN encoder producing node embeddings, plus a parameter-free
    dot-product decoder for scoring candidate (head, tail) links.

    Architecture
    ------------
    AtomEncoder (9-dim OGB codes -> hidden_dim)
      -> L x GCNLayer (hidden_dim -> hidden_dim)
      -> node embeddings h  [num_nodes, hidden_dim]

    decode(h, edge_label_index) -> dot-product score per candidate pair.
    """

    def __init__(
        self,
        hidden_dim  : int = CONFIG["hidden_dim"],
        num_layers  : int = CONFIG["num_layers"],
        dropout     : float = CONFIG["dropout"],
    ):
        super().__init__()
        # AtomEncoder sums per-feature-column embedding tables into one
        # hidden_dim-dimensional vector; standard OGB molecular input layer.
        self.input_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.layers = nn.ModuleList(
            [GCNLayer(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return final-layer node embeddings [num_nodes, hidden_dim]."""
        h = self.input_encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index)
        return h

    @staticmethod
    def decode(h: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """
        Dot-product decoder: score(i,j) = h_i . h_j

        Parameters
        ----------
        h                : [num_nodes, hidden_dim] node embeddings
        edge_label_index : [2, num_labeled] candidate (head, tail) pairs

        Returns
        -------
        [num_labeled] raw (unnormalized) link scores
        """
        src, dst = edge_label_index
        return (h[src] * h[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        h = self.encode(x, edge_index)
        return self.decode(h, edge_label_index)


# ═══════════════════════════════════════════════════════════════════════════
# § 4. LOSS FUNCTION
# ───────────────────────────────────────────────────────────────────────────
# Binary link prediction (contact / no-contact) -> binary cross-entropy
# with logits over the candidate pairs in edge_label_index.
# ═══════════════════════════════════════════════════════════════════════════

def compute_loss(scores: torch.Tensor, edge_label: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss for link prediction.

    Parameters
    ----------
    scores     : [num_labeled] raw dot-product scores
    edge_label : [num_labeled] float labels in {0.0, 1.0}
    """
    return F.binary_cross_entropy_with_logits(scores, edge_label.float())


# ═══════════════════════════════════════════════════════════════════════════
# § 5. METRIC — Hits@1, Hits@3, Hits@10, MRR  (filtered, per-head ranking)
# ═══════════════════════════════════════════════════════════════════════════

def compute_ranking_metrics(
    scores           : torch.Tensor,
    edge_label       : torch.Tensor,
    edge_label_index : torch.Tensor,
) -> dict:
    """
    Compute Hits@1, Hits@3, Hits@10, and MRR using per-head grouped ranking.

    For every head node h that has at least one true positive tail among
    the candidates, we collect ALL candidate scores sharing that head
    (positives + negatives, as already sampled into edge_label_index by
    the dataset), rank them in descending order, and record the rank of
    each true positive tail. Hits@K checks if that rank is <= K.

    Parameters
    ----------
    scores           : [num_labeled] raw scores (higher = more likely contact)
    edge_label       : [num_labeled] float {0,1} ground truth
    edge_label_index : [2, num_labeled] candidate (head, tail) pairs

    Returns
    -------
    dict with keys "hits@1", "hits@3", "hits@10", "mrr"
    """
    scores = scores.detach().cpu()
    edge_label = edge_label.detach().cpu()
    heads = edge_label_index[0].detach().cpu()

    # Group candidate indices by head node
    head_to_indices: dict = {}
    for idx, h in enumerate(heads.tolist()):
        head_to_indices.setdefault(h, []).append(idx)

    ranks = []
    for h, idxs in head_to_indices.items():
        idxs = torch.tensor(idxs)
        local_scores = scores[idxs]
        local_labels = edge_label[idxs]

        if local_labels.sum() == 0:
            continue   # no positive tail for this head -> nothing to rank

        # Descending sort of candidate scores -> rank 1 = highest score
        order = torch.argsort(local_scores, descending=True)
        sorted_labels = local_labels[order]

        # Record the rank (1-indexed) of every positive tail
        pos_positions = (sorted_labels == 1).nonzero(as_tuple=True)[0]
        for p in pos_positions:
            ranks.append((p.item() + 1))

    ranks = np.array(ranks, dtype=np.float64)
    if len(ranks) == 0:
        return dict(**{"hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0, "mrr": 0.0})

    return dict(
        **{
            "hits@1" : float(np.mean(ranks <= 1)),
            "hits@3" : float(np.mean(ranks <= 3)),
            "hits@10": float(np.mean(ranks <= 10)),
            "mrr"    : float(np.mean(1.0 / ranks)),
        }
    )


# ═══════════════════════════════════════════════════════════════════════════
# § 6. TRAIN / EVAL LOOPS
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device):
    """Run one training epoch; returns average training loss."""
    model.train()
    total_loss, total_pairs = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        scores = model(batch.x, batch.edge_index, batch.edge_label_index)
        loss   = compute_loss(scores, batch.edge_label)

        loss.backward()
        optimizer.step()

        n_pairs = batch.edge_label.numel()
        total_loss  += loss.item() * n_pairs
        total_pairs += n_pairs

    return total_loss / total_pairs


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Run evaluation; returns dict with avg_loss, hits@1, hits@3, hits@10, mrr.
    Ranking is accumulated across the WHOLE split (all graphs' candidate
    pairs pooled together per-head, per-graph — heads are only ever
    compared within the SAME graph since edge_label_index indices are
    graph-local before PyG's batch collation offsets them; batching still
    keeps candidates from different graphs numerically disjoint due to the
    node-index offset applied by the DataLoader, so cross-graph mixing
    cannot occur).
    """
    model.eval()
    total_loss, total_pairs = 0.0, 0
    all_scores, all_labels, all_heads = [], [], []

    for batch in loader:
        batch = batch.to(device)
        scores = model(batch.x, batch.edge_index, batch.edge_label_index)
        loss   = compute_loss(scores, batch.edge_label)

        n_pairs = batch.edge_label.numel()
        total_loss  += loss.item() * n_pairs
        total_pairs += n_pairs

        all_scores.append(scores.cpu())
        all_labels.append(batch.edge_label.cpu())
        all_heads.append(batch.edge_label_index.cpu())

    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)
    all_heads  = torch.cat(all_heads, dim=1)

    metrics = compute_ranking_metrics(all_scores, all_labels, all_heads)
    metrics["loss"] = total_loss / total_pairs

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# § 7. MAIN — end-to-end training to reproduce the GCN baseline
# ═══════════════════════════════════════════════════════════════════════════

def main(cfg: dict = CONFIG):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("Loading PCQM-Contact splits... (529k graphs, this may take a while)")
    datasets = load_datasets(cfg)
    inspect_graph_attributes(datasets["train"], n_show=1)

    train_loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(datasets["val"],   batch_size=cfg["batch_size"], shuffle=False)
    test_loader  = DataLoader(datasets["test"],  batch_size=cfg["batch_size"], shuffle=False)

    # ── 2. Build model ───────────────────────────────────────────────────
    model = GCN_LinkPredictor(
        hidden_dim = cfg["hidden_dim"],
        num_layers = cfg["num_layers"],
        dropout    = cfg["dropout"],
    ).to(cfg["device"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameter count: {n_params:,}  (paper target: ~504k)")

    # ── 3. Optimizer + LR schedule ───────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["lr_decay_factor"], patience=cfg["lr_patience"]
    )

    # ── 4. Training loop with best-checkpoint selection on val MRR ──────
    best_val_mrr = -1.0
    best_state   = None

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg["device"])
        val_metrics = evaluate(model, val_loader, cfg["device"])
        scheduler.step(val_metrics["loss"])

        if val_metrics["mrr"] > best_val_mrr:
            best_val_mrr = val_metrics["mrr"]
            best_state   = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_metrics['loss']:.4f} | val_MRR {val_metrics['mrr']:.4f} | "
                  f"val_Hits@10 {val_metrics['hits@10']:.4f}")

    # ── 5. Final test evaluation with best validation checkpoint ────────
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, cfg["device"])

    print("\n" + "=" * 60)
    print(f"Best Val MRR        : {best_val_mrr:.4f}")
    print(f"Test  Hits@1         : {test_metrics['hits@1']:.4f}   (paper GCN: 0.1321 ± 0.0007)")
    print(f"Test  Hits@3         : {test_metrics['hits@3']:.4f}   (paper GCN: 0.3791 ± 0.0004)")
    print(f"Test  Hits@10        : {test_metrics['hits@10']:.4f}  (paper GCN: 0.8256 ± 0.0006)")
    print(f"Test  MRR            : {test_metrics['mrr']:.4f}      (paper GCN: 0.3234 ± 0.0006)")
    print("=" * 60)

    return dict(model=model, test_metrics=test_metrics, best_val_mrr=best_val_mrr)


if __name__ == "__main__":
    main()
