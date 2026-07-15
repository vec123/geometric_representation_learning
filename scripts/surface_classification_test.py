"""Sanity test: classify primitive shapes from the encoder's INVARIANT scalar latent.

Shapes are the point clouds in Dataset/Primitives/*.vtp (box, ellipse, pyramid, sphere),
each a surface sampled at discrete vertices. Each vertex carries:
  * position          (fed geometrically, not as a feature)
  * unit normal        -> a ``1o`` equivariant input feature
  * relative area       -> a ``0e`` invariant input feature (local sampling density)

The encoder runs its equivariant GNN on the full vertex graph, then AGGREGATES to a small
set of supernodes (via the shared src.learning.helpers.build_training_graph that
equivariant_gnn_train also uses), and the SE(3) transformer + scalar readout act on
those supernodes -- so the O(N^2) attention is over ~16 supernodes, not the ~512 vertices
(keeps it comfortably inside 8 GB VRAM). A small MLP head is attached to the encoder's
``mu`` (the SE(3)-invariant scalar latent) and trained with cross-entropy.

We then check three things the equivariant design should give us:
  1. Accuracy        -- surfaces are classifiable from the invariant latent.
  2. Rotation/translation invariance -- a shape and a randomly rotated+translated copy give
     the same latent and prediction, with NO augmentation.
  3. Resampling stability -- the same surface at a different vertex sampling gives a nearby
     latent. Sampling is modelled as a DROPOUT of one fixed canonical (high) resolution
     (as in equivariant_gnn_train), and supernodes are FPS-sampled on that canonical cloud
     so they don't move between resamplings -- the remaining drift is only the area-weighted
     neighbourhood each supernode aggregates.

Run:  python scripts/surface_classification_test.py
"""

import os
import sys
import math
import json
import argparse

os.environ.setdefault("MPLBACKEND", "Agg")   # headless-safe if anything ever plots

import numpy as np
import torch
import torch.nn as nn
from e3nn import o3

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.learning.models.group_encoder import GroupEncoder
from src.learning.logger.headless import enable_headless
from src.learning.helpers import build_training_graph
from src.vtk.io import load_vtp
from src.vtk.extract import extract_vtp_points_cells, extract_vtp_point_fields
from src.learning.logger.train_logs import TrainingLogger

# --------------------------------------------------------------------------- #
# Primitive shapes:  Each file is a 512-point cloud carrying two point fields:
#   * ``normal`` [N,3] -- unit, outward-oriented     -> a ``1o`` equivariant input feature
#   * ``area``   [N]   -- per-vertex surface measure -> a ``0e`` invariant input feature
# There is ONE canonical cloud per class; training variety comes from random rotations/
# translations (make_sample) and vertex dropout/resampling (build_batch) -- 
# exactly the augmentations
# the invariance [2] and resampling-stability [3] probes are built to test.
# --------------------------------------------------------------------------- #

PRIMITIVES_DIR = os.path.join(ROOT, 'Dataset', 'Primitives')
PRIMITIVE_FILES = {
    'box':     'box.vtp',
    'ellipse': 'ellipse.vtp',
    'pyramid': 'pyramid.vtp',
    'sphere':  'sphere.vtp',
}
CLASSES = list(PRIMITIVE_FILES)

_PRIM_CACHE = {}

# --------------------------------------------------------------------------- #
R_MAX          = 0.22    # full-graph radius ball 
R_SUPERGRAPH   = 0.5    # bipartite radius
N_SUPERNODES = 10
SUPERNODE_MODE = 'fps'   # supernode sampling, as in equivariant_gnn_train's sample_nodes:
                         # 'fps' -> deterministic, rotation/translation-invariant selection


def _load_primitive(name):
    """Load a primitive -> (pos [N,3], unit normal [N,3], rel_area [N,1]) as torch.float32.

    The .vtp already carries unit outward normals and per-vertex ``area`` as point fields
    (Dataset/add_point_features.py), so nothing geometric is recomputed here. ``rel_area``
    is the area field divided by its mean -- a scale-free local sampling density (O(1)) that
    matches the 0e input-feature scale; the encoder's area pooling is itself scale-invariant,
    so build_batch reuses the same normalised area for the supernode Voronoi mass. Cached:
    each file is read from disk once."""
    if name not in _PRIM_CACHE:
        path = os.path.join(PRIMITIVES_DIR, PRIMITIVE_FILES[name])
        polydata = load_vtp(path)
        points, _ = extract_vtp_points_cells(polydata)
        fields = extract_vtp_point_fields(polydata, ['area', 'normal'])
        area, normal = fields['area'], fields['normal']
        if area is None or normal is None:
            raise ValueError(f"{path} is missing 'area'/'normal' point fields -- run "
                             f"Dataset/add_point_features.py on the primitives first.")
        pos = torch.tensor(np.asarray(points), dtype=torch.float32)
        normal = torch.tensor(np.asarray(normal), dtype=torch.float32)
        area = torch.tensor(np.asarray(area), dtype=torch.float32)
        rel_area = (area / area.mean()).unsqueeze(-1)
        _PRIM_CACHE[name] = (pos, normal, rel_area)
    pos, normal, rel_area = _PRIM_CACHE[name]
    return pos.clone(), normal.clone(), rel_area.clone()


def make_sample(name, rotate=True):
    """(pos [N,3], x [N,5], area [N], y int).  x = [1, rel_area, normal] -> '2x0e + 1x1o'.
    ``area`` is the per-vertex surface measure (rotation/translation invariant), used for
    area-weighted pooling. The canonical geometry is fixed per class (one .vtp); ``rotate``
    only draws the random rotation+translation that gives each sample a different pose."""
    pos, normal, rel_area = _load_primitive(name)
    if rotate:
        R, t = o3.rand_matrix(), torch.randn(3)
        pos = pos @ R.T + t
        normal = normal @ R.T                                 # normals are 1o -> rotate too
    x = torch.cat([torch.ones(pos.shape[0], 1), rel_area, normal], dim=-1)
    scalar_feats = torch.ones(pos.shape[0], 1)
    return pos, scalar_feats, normal, rel_area.squeeze(-1), CLASSES.index(name)


def build_batch(samples, S=16, r_max=R_MAX, r_super=R_SUPERGRAPH,
                super_mode=SUPERNODE_MODE, dropout=0.0, device='cpu', key=None,
                bipartite_mode='radius', bipartite_seed=0, recompute_area=True):
    """Assemble a list of canonical (pos, x, area, y) primitives into (graph, supergraph, y)
    via the shared build_training_graph (see the module comment above for the design).

    ``dropout`` is the fraction of canonical vertices dropped to form the current graph;
    a "resampling" is two draws with (possibly) different dropout. ``key`` (a torch.Generator)
    makes the dropout reproducible. Every primitive has the same vertex count, so the shapes
    stack directly into the padded [B, N, *] tensors build_training_graph expects (all-true
    mask -> no real padding).

    ``bipartite_mode`` selects supernode aggregation: 'radius' (each supernode gathers its
    r_super ball, capping the denser ones at 128 via a SEEDED random subset -> reproducible
    'keep 128 at random') or 'voronoi' (each vertex -> its single nearest supernode, a
    partition with no cap). ``bipartite_seed`` seeds the radius-mode neighbour subsampling."""
    verts = torch.stack([s[0] for s in samples])              # [B, N, 3] positions
    scalar_feats = torch.stack([s[1] for s in samples])   # only scalar invariants
    normals = torch.stack([s[2] for s in samples])        # vector features
    areas = torch.stack([s[3] for s in samples])  

    B, N = verts.shape[:2]
    mask = torch.ones(B, N, dtype=torch.bool)
    y = torch.tensor([s[4] for s in samples], dtype=torch.long)

    voronoi = (bipartite_mode == 'voronoi')
    graph, supergraph = build_training_graph(
        verts, mask, key,
        r_max=r_max, dropout_rate=dropout,
        n_supernodes=S, r_supergraph=r_super, use_supernodes=True,
        sampling_mode_graph='uniform', sampling_mode_supernodes=super_mode,
        features=scalar_feats,
        areas=areas, 
        normals=normals,
        bipartite_voronoi=voronoi,
        bipartite_seed=(None if voronoi else bipartite_seed),
        bipartite_max_neighbors=1024,
        recompute_area = recompute_area)
    return graph.to(device), supergraph.to(device), y.to(device)


# --------------------------------------------------------------------------- #
# Model: encoder scalar latent (mu, over supernodes) -> classification head
# --------------------------------------------------------------------------- #
class SurfaceClassifier(nn.Module):
    def __init__(self, num_classes, latent_dim=12, transformer_type='se3', area_pool=True):
        super().__init__()
        # Slim (fits ~8 GB): narrow irreps, sh_lmax=1, transformer over the SUPERNODES.
        cfg = {
            'input_irreps': '1x0e + 1x1o',
            'intermediate_irreps': '16x0e + 8x1o',
            'output_irreps': f'{latent_dim}x0e + 2x1o',
        }
        tcfg = {'num_layers': 1, 'num_heads': 2, 'hidden_channels': 8, 'sh_lmax': 1}
        self.encoder = GroupEncoder(latent_dim=latent_dim, irreps_cfg=cfg, sh_lmax=1,
                                    readout='mean', supernode_sh_lmax=2,
                                    transformer_type=transformer_type,
                                    transformer_cfg=tcfg, area_pool=area_pool, verbose=False)
        self.head = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(),
                                  nn.Linear(32, num_classes))

    def encode(self, graph, supergraph):
        return self.encoder(graph, supergraph).mu             # [B, latent_dim], invariant

    def forward(self, graph, supergraph):
        return self.head(self.encode(graph, supergraph))


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def build_split(n_per_class, rng, rotate=True):
    data = [make_sample(name, rotate=rotate)
            for name in CLASSES for _ in range(n_per_class)]
    rng.shuffle(data)
    return data


@torch.no_grad()
def accuracy(model, data, device, bs=16, gcfg=None):
    model.eval()
    gcfg = gcfg or {}
    correct = total = 0
    for i in range(0, len(data), bs):
        g, sg, y = build_batch(data[i:i + bs], 
                               device=device,
                                recompute_area=True,
                                **gcfg)
        correct += (model(g, sg).argmax(-1) == y).sum().item(); total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser(description="Surface-classification sanity test for the "
                                             "equivariant encoder's scalar latent.")
    ap.add_argument("--remote", action="store_true",
                    help="Headless/HPC mode: timestamped, flushed logs + a JSON metrics "
                         "summary in --log-dir. Auto-enabled when stdout is not a TTY.")
    ap.add_argument("--local", action="store_true",
                    help="Force interactive mode (disable the not-a-TTY auto-remote).")
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--area-pool", dest="area_pool", action="store_true", default=True,
                    help="Area-weighted pooling (default on): makes the latent a surface "
                         "integral, so resampling drifts it less.")
    ap.add_argument("--no-area-pool", dest="area_pool", action="store_false",
                    help="Disable area weighting (uniform pooling) for a before/after check.")
    # Graph-construction knobs (the main levers on resampling stability [3]): more
    # supernodes + a wider bipartite radius make the area-weighted pool a lower-variance
    # estimate of the surface integral, so which supernodes FPS happens to pick matters less.
    ap.add_argument("--supernodes", type=int, default=N_SUPERNODES, help="n supernodes S (sample_nodes).")
    ap.add_argument("--r-max", type=float, default=R_MAX, help="full-graph radius ball.")
    ap.add_argument("--r-super", type=float, default=R_SUPERGRAPH,
                    help="bipartite radius: neighbourhood each supernode aggregates over.")
    ap.add_argument("--super-mode", default=SUPERNODE_MODE, choices=["fps", "uniform", "gaussian"],
                    help="supernode sampling mode (as in equivariant_gnn_train).")
    ap.add_argument("--dropout", type=float, default=0.4,
                    help="fraction of canonical vertices dropped to form the current graph "
                         "each step (resampling = dropout of the highest resolution).")
    ap.add_argument("--consistency-weight", dest="consistency_weight", type=float, default=0.0,
                    help="two-view resampling-consistency loss (as in equivariant_gnn_train's "
                         "CONTRASTIVE path): >0 draws a 2nd dropout of the same canonical cloud "
                         "each step and pulls the two latents together -> directly trains down [3].")
    ap.add_argument("--bipartite-mode", default="radius", choices=["radius", "voronoi"],
                    help="supernode aggregation (build_bipartite_graph): 'radius' -> each "
                         "supernode gathers its r_super ball, capping dense ones at 128 via a "
                         "SEEDED random subset; 'voronoi' -> each vertex to its single nearest "
                         "supernode (a partition, no cap).")
    ap.add_argument("--bipartite-seed", type=int, default=0,
                    help="seed for the radius-mode 128-neighbour random subsampling (reproducible "
                         "'keep 128 at random'); unused in voronoi mode.")
    args = ap.parse_args()

    # Toggle: --remote / --local, else auto (headless when stdout is not a TTY).
    remote = True if args.remote else (False if args.local else None)
    is_remote, log_path = enable_headless(args.log_dir, remote=remote, name="surface_test")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    key = torch.Generator().manual_seed(args.seed)            # reproducible dropout
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    S, BS = args.supernodes, 16
    NVERT = _load_primitive(CLASSES[0])[0].shape[0]           # canonical verts per shape (512)

    # Shared graph-construction config, threaded into every build_batch call so train,
    # eval and the invariance probes all build supernodes the same way.
    gcfg = dict(S=S, r_max=args.r_max, r_super=args.r_super, super_mode=args.super_mode,
                bipartite_mode=args.bipartite_mode, bipartite_seed=args.bipartite_seed)

    print(f"mode={'REMOTE/headless' if is_remote else 'local'}  device={device}  "
          f"classes={CLASSES}  canon_verts~{NVERT}  supernodes={S}  epochs={args.epochs}  seed={args.seed}  "
          f"area_pool={args.area_pool}  r_max={args.r_max}  r_super={args.r_super}  super_mode={args.super_mode}  "
          f"dropout={args.dropout}  bipartite={args.bipartite_mode}")

    train = build_split(50, rng, rotate=True)
    test  = build_split(15, rng, rotate=True)

    model = SurfaceClassifier(num_classes=len(CLASSES), latent_dim=12,
                              area_pool=args.area_pool).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    lossfn = nn.CrossEntropyLoss()

    logger = TrainingLogger(log_dir="surface_classification")

    for ep in range(args.epochs):
        model.train(); rng.shuffle(train); tot = 0.0
        for i in range(0, len(train), BS):
            batch = train[i:i + BS]
            g, sg, y = build_batch(batch, device=device, dropout=args.dropout, key=key, **gcfg)

            if ep %10 == 0  and i==0:
                logger.visualize_batch( (g,sg, None,None), 
                                    None, 
                                    step=ep, 
                                    subdir = "vtk", max_num= 4)
            
            mu_a = model.encode(g, sg)
            loss = lossfn(model.head(mu_a), y)
            # Two-view resampling-consistency: a 2nd dropout of the SAME canonical clouds
            # (same fixed supernodes) -> classify it too and pull the two latents together.
            if args.consistency_weight > 0:
                gb, sgb, _ = build_batch(batch, device=device, dropout=args.dropout, key=key, **gcfg)
                mu_b = model.encode(gb, sgb)
                loss = loss + lossfn(model.head(mu_b), y)
                loss = loss + args.consistency_weight * ((mu_a - mu_b) ** 2).mean()

            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * y.numel()
        msg = f"epoch {ep:3d}/{args.epochs}  loss {tot/len(train):.4f}"
        if ep % 5 == 0 or ep == args.epochs - 1:
            msg += f"  test_acc {accuracy(model, test, device, gcfg=gcfg):.3f}"
        print(msg)

    acc = accuracy(model, test, device, gcfg=gcfg)
    print(f"[1] FINAL test accuracy: {acc:.3f}")

    # [2] Rotation/translation invariance, split into two questions:
    #   [2a] Is the ENCODER invariant? Rotate the SAME graph (identical supernodes/edges,
    #        only positions + normals move) -> should be exact to float tolerance.
    #   [2b] Is the END-TO-END pipeline invariant? Rebuild the graph from the transformed
    #        cloud -> exposes graph-construction sensitivity (FPS supernode tie-breaking on
    #        symmetric shapes), which perturbs the latent but not, in practice, the class.
    probe = [make_sample(name, rotate=False) for name in CLASSES for _ in range(10)]
    g0, sg0, _ = build_batch(probe, device=device, **gcfg)

    R = o3.rand_matrix().to(device); t = torch.randn(3, device=device)
    g1, sg1 = g0.clone(), sg0.clone()
    g1.pos = g0.pos @ R.T + t
    g1.x = g0.x.clone()
    if hasattr(g0, 'normal') and g0.normal is not None:
        g1.normal = g0.normal @ R.T
    sg1.pos = sg0.pos @ R.T + t
    with torch.no_grad():
        mu0 = model.encode(g0, sg0)
        enc_inv = (mu0 - model.encode(g1, sg1)).abs().max().item()
    print(f"[2a] encoder invariance (fixed graph): max|mu - mu(Rx+t)| = {enc_inv:.2e}")

    rebuilt = []
    for pos, scalar_feats, normals, area, y in probe:
        Ri, ti = o3.rand_matrix(), torch.randn(3)
        normals2 = normals.clone() @ Ri.T
        rebuilt.append((pos @ Ri.T + ti, scalar_feats, normals2, area, y))
    g2, sg2, _ = build_batch(rebuilt, device=device, **gcfg)
    with torch.no_grad():
        mu2 = model.encode(g2, sg2)
        e2e_inv = (mu0 - mu2).abs().max().item()
        rot_agree = (model.head(mu0).argmax(-1) == model.head(mu2).argmax(-1)).float().mean().item()
    print(f"[2b] end-to-end (graph rebuilt via FPS): max|mu diff| = {e2e_inv:.2e} "
             f"| prediction agreement = {rot_agree*100:.1f}%")

    # [3] Resampling stability, in the canonical-dropout model: ONE canonical cloud per
    # primitive, two DIFFERENT dropouts of it. Supernodes are FPS-sampled on the shared
    # canonical cloud, so both views get the IDENTICAL supernode set -- only the current
    # (dropped) vertices each supernode aggregates differ. This is the design under test:
    # drift should come only from the area-weighted neighbourhood, not from a moving
    # supernode set (which was the dominant source before). Geometry is fixed per class
    # (one .vtp), so the 8 repeats just average over the random dropout realisations.
    res_key = torch.Generator().manual_seed(7)
    drifts, res_agree = [], []
    for name in CLASSES:
        pos, nrm, rel_area = _load_primitive(name)                     # canonical cloud (fixed per class)
        scalar_feats = torch.ones(pos.shape[0], 1)
        canon = (pos, scalar_feats , nrm,  rel_area.squeeze(-1), CLASSES.index(name))
        for _ in range(8):
            ga, sga, _ = build_batch([canon], device=device, dropout=0.3, key=res_key, **gcfg)
            gb, sgb, _ = build_batch([canon], device=device, dropout=0.6, key=res_key, **gcfg)
            with torch.no_grad():
                ma, mb = model.encode(ga, sga), model.encode(gb, sgb)
            drifts.append(((ma - mb).norm() / (ma.norm() + 1e-9)).item())
            res_agree.append(int(model.head(ma).argmax() == model.head(mb).argmax()))
    print(f"[3] resampling stability: mean relative latent drift = {np.mean(drifts):.3f} "
             f"| class agreement = {100*np.mean(res_agree):.1f}%")

    summary = {
        "mode": "remote" if is_remote else "local",
        "device": device, "seed": args.seed, "epochs": args.epochs, "classes": CLASSES,
        "area_pool": args.area_pool,
        "supernodes": S, "r_max": args.r_max, "r_super": args.r_super,
        "super_mode": args.super_mode, "dropout": args.dropout,
        "bipartite_mode": args.bipartite_mode, "bipartite_seed": args.bipartite_seed,
        "test_accuracy": acc,
        "encoder_invariance_max_mu_diff": enc_inv,
        "end_to_end_invariance_max_mu_diff": e2e_inv,
        "rotation_prediction_agreement": rot_agree,
        "resampling_mean_rel_drift": float(np.mean(drifts)),
        "resampling_class_agreement": float(np.mean(res_agree)),
    }
    if is_remote and log_path:
        json_path = (log_path[:-4] if log_path.endswith(".log") else log_path) + ".json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote metrics summary -> {json_path}")
    else:
        print("summary: " + json.dumps(summary))


if __name__ == '__main__':
    main()
