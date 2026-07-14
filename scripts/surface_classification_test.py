"""Sanity test: classify smooth surfaces from the encoder's INVARIANT scalar latent.

Surfaces are continuous 2-manifolds sampled at discrete vertices. Each vertex carries:
  * position          (fed geometrically, not as a feature)
  * unit normal        -> a ``1o`` equivariant input feature
  * relative area       -> a ``0e`` invariant input feature (local sampling density)

The encoder runs its equivariant GNN on the full vertex graph, then AGGREGATES to a small
set of supernodes (sampled + bipartite conv, via the shared src.graphs.graphs builders that
equivariant_gnn_train also uses), and the SE(3) transformer + scalar readout act on
those supernodes -- so the O(N^2) attention is over ~16 supernodes, not ~120 vertices
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
from src.graphs.graphs import build_radius_graph, sample_nodes, build_bipartite_graph


# --------------------------------------------------------------------------- #
# Parametric surfaces  S(u, v) -> R^3  (each returns [nu, nv, 3])
# --------------------------------------------------------------------------- #
def _sphere(U, V, p):
    r = p['r']
    return np.stack([r * np.sin(U) * np.cos(V), r * np.sin(U) * np.sin(V), r * np.cos(U)], -1)

def _torus(U, V, p):
    R, r = p['R'], p['r']
    return np.stack([(R + r * np.cos(V)) * np.cos(U),
                     (R + r * np.cos(V)) * np.sin(U), r * np.sin(V)], -1)

def _cylinder(U, V, p):
    r, h = p['r'], p['h']
    return np.stack([r * np.cos(U), r * np.sin(U), h * V], -1)

def _saddle(U, V, p):
    a = p['a']
    return np.stack([U, V, a * (U ** 2 - V ** 2)], -1)

SURFACES = {
    'sphere':   (_sphere,   (0.15, math.pi - 0.15), (0.0, 2 * math.pi)),
    'torus':    (_torus,    (0.0, 2 * math.pi),     (0.0, 2 * math.pi)),
    'cylinder': (_cylinder, (0.0, 2 * math.pi),     (-1.0, 1.0)),
    'saddle':   (_saddle,   (-1.0, 1.0),            (-1.0, 1.0)),
}
CLASSES = list(SURFACES)


def _random_params(name, rng):
    if name == 'sphere':   return {'r': rng.uniform(0.8, 1.4)}
    if name == 'torus':    return {'R': rng.uniform(1.0, 1.5), 'r': rng.uniform(0.3, 0.5)}
    if name == 'cylinder': return {'r': rng.uniform(0.5, 0.9), 'h': rng.uniform(0.8, 1.4)}
    if name == 'saddle':   return {'a': rng.uniform(0.6, 1.2)}
    raise KeyError(name)


def _geometry(name, nu, nv, p):
    """-> pos [N,3], unit normal [N,3], relative area [N,1] (all torch.float32)."""
    fn, (u0, u1), (v0, v1) = SURFACES[name]
    U, V = np.meshgrid(np.linspace(u0, u1, nu), np.linspace(v0, v1, nv), indexing='ij')
    P = fn(U, V, p)                                            # [nu, nv, 3]
    du, dv = np.gradient(P, axis=0), np.gradient(P, axis=1)    # tangents (finite diff)
    n = np.cross(du, dv)
    area = np.linalg.norm(n, axis=-1) + 1e-12                 # |S_u x S_v| ~ area element
    n = n / area[..., None]
    P, n, area = P.reshape(-1, 3), n.reshape(-1, 3), area.reshape(-1)
    c = P.mean(0, keepdims=True)                              # orient normals outward
    s = np.sign((n * (P - c)).sum(-1, keepdims=True)); s[s == 0] = 1.0
    n = n * s
    return (torch.tensor(P, dtype=torch.float32),
            torch.tensor(n, dtype=torch.float32),
            torch.tensor(area / area.mean(), dtype=torch.float32).unsqueeze(-1))


def make_sample(name, nu, nv, rng, rotate=True):
    """(pos [N,3], x [N,5], area [N], y int).  x = [1, rel_area, normal] -> '2x0e + 1x1o'.
    ``area`` is the per-vertex surface measure (rotation/translation invariant), used for
    area-weighted pooling. (For a real .vtp point cloud, replace _geometry's analytic area
    with mesh areas from src.preprocessing.SurfaceTriangulator.)"""
    pos, normal, rel_area = _geometry(name, nu, nv, _random_params(name, rng))
    if rotate:
        R, t = o3.rand_matrix(), torch.randn(3)
        pos = pos @ R.T + t
        normal = normal @ R.T                                 # normals are 1o -> rotate too
    x = torch.cat([torch.ones(pos.shape[0], 1), rel_area, normal], dim=-1)
    return pos, x, rel_area.squeeze(-1), CLASSES.index(name)


# --------------------------------------------------------------------------- #
# Graph construction -- reuses the shared src.graphs.graphs primitives that
# equivariant_gnn_train also builds its graphs with:
#   * build_radius_graph    -> full vertex graph (radius ball, not kNN)
#   * sample_nodes          -> supernode subset (fps | uniform | gaussian)
#   * build_bipartite_graph -> supernode<-vertex aggregation edges (radius)
#
# Canonical-resolution design (as in equivariant_gnn_train's build_super_graph +
# ResamplingGraphLoader): each ``samples`` entry is the surface at a FIXED, high
# "canonical" resolution. Supernodes are FPS-sampled on that canonical cloud (so the
# supernode set is resolution-INDEPENDENT), and their Voronoi mass is measured on it
# too. The full graph fed to the GNN is a DROPOUT of the canonical vertices -- i.e. a
# "resampling" is just a dropout of the highest resolution -- and the bipartite edges
# connect the fixed canonical supernodes to whichever current vertices survive. This
# is what keeps [3] stable: two resamplings share the exact same supernodes, so only
# the (area-weighted) neighbourhood each supernode aggregates changes.
#
# Per-vertex features (x = const, area, normal) and the area attributes are surface-
# specific and assembled here -- the shared builders carry positions/edges/batch only.
# --------------------------------------------------------------------------- #
R_MAX          = 0.7     # full-graph radius ball (replaces the old kNN k_full)
R_SUPERGRAPH   = 1.2     # bipartite radius: neighbourhood each supernode aggregates over
SUPERNODE_MODE = 'fps'   # supernode sampling, as in equivariant_gnn_train's sample_nodes:
                         # 'fps' -> deterministic, rotation/translation-invariant selection
                         # (keeps the [2b]/[3] invariance checks meaningful); 'uniform' or
                         # 'gaussian' match the other modes but are random.


def build_batch(samples, S=16, r_max=R_MAX, r_super=R_SUPERGRAPH,
                super_mode=SUPERNODE_MODE, dropout=0.0, device='cpu', key=None):
    """Assemble a list of CANONICAL (high-res) (pos, x, area, y) into (graph, supergraph, y).

    graph      : full vertex graph over a DROPOUT of the canonical vertices (radius ball,
                 build_radius_graph) with the surviving x, pos, batch and per-vertex ``area``.
    supergraph : supernodes FPS-sampled on the CANONICAL cloud (fixed across dropouts) +
                 bipartite radius edges to the current vertices (build_bipartite_graph;
                 row0=super target, row1=full source), plus per-supernode ``area`` (Voronoi
                 mass measured on the canonical cloud). The ``area`` attributes feed
                 GroupEncoder's area-weighted pooling (area_pool=True).

    ``dropout`` is the fraction of canonical vertices dropped to form the current graph;
    a "resampling" is two draws with (possibly) different dropout. ``key`` (a
    torch.Generator) makes the dropout reproducible.
    """
    xs, ps, av, fb, ys = ([] for _ in range(5))
    sp, sb, sa = [], [], []
    for b, (pos, x, area, y) in enumerate(samples):
        Nc = pos.shape[0]                                     # canonical vertex count

        # --- supernodes on the CANONICAL cloud (fixed, resolution-independent) ---
        spos, _ = sample_nodes(pos.unsqueeze(0), torch.ones(1, Nc, dtype=torch.bool),
                               num_samples=S, mode=super_mode, key=key)
        s = spos.shape[0]
        sp.append(spos); sb.append(torch.full((s,), b, dtype=torch.long))
        # supernode mass: nearest-supernode Voronoi partition of the canonical surface
        # (measured on the full cloud -> identical for every dropout of it).
        nearest = torch.cdist(spos, pos).argmin(dim=0)        # [Nc] nearest supernode
        A_s = torch.zeros(s); A_s.index_add_(0, nearest, area)
        sa.append(A_s)

        # --- current graph = dropout of the canonical vertices ---
        if dropout > 0:
            keep = torch.rand(Nc, generator=key) > dropout
            if int(keep.sum()) < 3:                           # never empty a shape
                keep = torch.ones(Nc, dtype=torch.bool)
        else:
            keep = torch.ones(Nc, dtype=torch.bool)
        cpos, cx, carea = pos[keep], x[keep], area[keep]
        N = cpos.shape[0]
        xs.append(cx); ps.append(cpos); av.append(carea); ys.append(y)
        fb.append(torch.full((N,), b, dtype=torch.long))

    full_pos, full_batch = torch.cat(ps), torch.cat(fb)
    # full graph: shared radius-graph builder (batch-aware -> no cross-shape edges).
    graph = build_radius_graph(full_pos, full_batch, r_max=r_max)
    graph.x, graph.area = torch.cat(xs), torch.cat(av)

    # supernode aggregation: shared radius bipartite builder (row0=super, row1=full).
    supergraph = build_bipartite_graph(full_pos, full_batch,
                                       torch.cat(sp), torch.cat(sb), r_max=r_super)
    supergraph.area = torch.cat(sa)

    y = torch.tensor(ys, dtype=torch.long)
    return graph.to(device), supergraph.to(device), y.to(device)


# --------------------------------------------------------------------------- #
# Model: encoder scalar latent (mu, over supernodes) -> classification head
# --------------------------------------------------------------------------- #
class SurfaceClassifier(nn.Module):
    def __init__(self, num_classes, latent_dim=12, transformer_type='se3', area_pool=True):
        super().__init__()
        # Slim (fits ~8 GB): narrow irreps, sh_lmax=1, transformer over the SUPERNODES.
        cfg = {
            'input_irreps': '2x0e + 1x1o',
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
def build_split(n_per_class, nu, nv, rng, rotate=True):
    data = [make_sample(name, nu, nv, rng, rotate=rotate)
            for name in CLASSES for _ in range(n_per_class)]
    rng.shuffle(data)
    return data


@torch.no_grad()
def accuracy(model, data, device, bs=16, gcfg=None):
    model.eval()
    gcfg = gcfg or {}
    correct = total = 0
    for i in range(0, len(data), bs):
        g, sg, y = build_batch(data[i:i + bs], device=device, **gcfg)
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
    ap.add_argument("--epochs", type=int, default=25)
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
    ap.add_argument("--supernodes", type=int, default=16, help="n supernodes S (sample_nodes).")
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
    args = ap.parse_args()

    # Toggle: --remote / --local, else auto (headless when stdout is not a TTY).
    remote = True if args.remote else (False if args.local else None)
    is_remote, log_path = enable_headless(args.log_dir, remote=remote, name="surface_test")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    key = torch.Generator().manual_seed(args.seed)            # reproducible dropout
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    NU, NV, S, BS = 18, 15, args.supernodes, 16               # ~270 CANONICAL verts, S supernodes

    # Shared graph-construction config, threaded into every build_batch call so train,
    # eval and the invariance probes all build supernodes the same way.
    gcfg = dict(S=S, r_max=args.r_max, r_super=args.r_super, super_mode=args.super_mode)

    print(f"mode={'REMOTE/headless' if is_remote else 'local'}  device={device}  "
          f"classes={CLASSES}  canon_verts~{NU*NV}  supernodes={S}  epochs={args.epochs}  seed={args.seed}  "
          f"area_pool={args.area_pool}  r_max={args.r_max}  r_super={args.r_super}  super_mode={args.super_mode}  "
          f"dropout={args.dropout}")

    train = build_split(50, NU, NV, rng, rotate=True)
    test  = build_split(15, NU, NV, rng, rotate=True)

    model = SurfaceClassifier(num_classes=len(CLASSES), latent_dim=12,
                              area_pool=args.area_pool).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    lossfn = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        model.train(); rng.shuffle(train); tot = 0.0
        for i in range(0, len(train), BS):
            batch = train[i:i + BS]
            g, sg, y = build_batch(batch, device=device, dropout=args.dropout, key=key, **gcfg)
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
    inv_rng = np.random.default_rng(123)
    probe = [make_sample(name, NU, NV, inv_rng, rotate=False) for name in CLASSES for _ in range(10)]
    g0, sg0, _ = build_batch(probe, device=device, **gcfg)

    R = o3.rand_matrix().to(device); t = torch.randn(3, device=device)
    g1, sg1 = g0.clone(), sg0.clone()
    g1.pos = g0.pos @ R.T + t
    g1.x = g0.x.clone(); g1.x[:, 2:5] = g0.x[:, 2:5] @ R.T     # rotate the normal channels
    sg1.pos = sg0.pos @ R.T + t
    with torch.no_grad():
        mu0 = model.encode(g0, sg0)
        enc_inv = (mu0 - model.encode(g1, sg1)).abs().max().item()
    print(f"[2a] encoder invariance (fixed graph): max|mu - mu(Rx+t)| = {enc_inv:.2e}")

    rebuilt = []
    for pos, x, area, y in probe:
        Ri, ti = o3.rand_matrix(), torch.randn(3)
        x2 = x.clone(); x2[:, 2:5] = x[:, 2:5] @ Ri.T
        rebuilt.append((pos @ Ri.T + ti, x2, area, y))
    g2, sg2, _ = build_batch(rebuilt, device=device, **gcfg)
    with torch.no_grad():
        mu2 = model.encode(g2, sg2)
        e2e_inv = (mu0 - mu2).abs().max().item()
        rot_agree = (model.head(mu0).argmax(-1) == model.head(mu2).argmax(-1)).float().mean().item()
    print(f"[2b] end-to-end (graph rebuilt via FPS): max|mu diff| = {e2e_inv:.2e} "
             f"| prediction agreement = {rot_agree*100:.1f}%")

    # [3] Resampling stability, in the canonical-dropout model: ONE canonical cloud per
    # surface, two DIFFERENT dropouts of it. Supernodes are FPS-sampled on the shared
    # canonical cloud, so both views get the IDENTICAL supernode set -- only the current
    # (dropped) vertices each supernode aggregates differ. This is the design under test:
    # drift should now come only from the area-weighted neighbourhood, not from a moving
    # supernode set (which was the dominant source before).
    res_rng = np.random.default_rng(7)
    res_key = torch.Generator().manual_seed(7)
    drifts, res_agree = [], []
    for name in CLASSES:
        for _ in range(8):
            p = _random_params(name, res_rng)
            pos, nrm, ar = _geometry(name, NU, NV, p)              # canonical high-res cloud
            x = torch.cat([torch.ones(pos.shape[0], 1), ar, nrm], -1)
            canon = (pos, x, ar.squeeze(-1), CLASSES.index(name))
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
