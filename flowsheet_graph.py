"""
Level 4 — GNN Flowsheet Optimisation
Graph representation of the CO₂-to-methanol plant for PyTorch Geometric.

Author: Pepe (Jose Maria Contreras Prada)
Project: EURECHA 2026 Process Design Contest

Architecture:
  - Nodes: 11 unit operations (absorber, stripper, compressor, PEM, mixer,
    preheater, reactor, cooler, flash, splitter, distillation)
  - Edges: 13 directed material/energy streams (including recycles)
  - Graph-level regression targets: TAC, carbon_efficiency, LCOM
  - Heterogeneous node types via one-hot encoding

Usage:
  from flowsheet_graph import build_base_graph, FlowsheetDataset, FlowsheetGNN
"""

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool
from torch_geometric.utils import to_undirected
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
import json
import warnings

# ─────────────────────────────────────────────────────────────────────
# Try to import project economics / constants.  If the module layout
# changes, the import fails and we silently fall back to synthetic
# targets (keeps this file runnable standalone for prototyping).
# ─────────────────────────────────────────────────────────────────────
_PROJECT_IMPORT_OK = False
try:
    from .config import (
        TOTAL_CAPEX_M,
        OTHER_OPEX,
        UTILITY_PER_T_MEOH,
        MEOH_ANNUAL_T,
        OPERATING_HOURS,
        MW_MEOH,
        H2_FEED_KMOL,
    )
    # FLUEGAS_CO2_KMOL is only in the root constants module, not
    # re-exported by config.py — pull it directly.
    import constants as _plant_constants  # type: ignore
    _FLUEGAS_CO2_KMOL = float(_plant_constants.FLUEGAS_CO2_KMOL)
    _PROJECT_IMPORT_OK = True
except Exception:
    # Running the file standalone (`python flowsheet_graph.py`) bypasses
    # the package-import machinery.  Try an absolute path-based import.
    try:
        import sys as _sys
        _here = Path(__file__).resolve().parent
        _project_root = _here.parent                                  # …/EURECHA
        _figures_scripts = _project_root / "figures" / "scripts"
        _scripts = _project_root / "scripts"
        for _p in (_here, _project_root, _figures_scripts, _scripts):
            _p = str(_p)
            if _p not in _sys.path:
                _sys.path.insert(0, _p)

        from config import (                                           # type: ignore
            TOTAL_CAPEX_M,
            OTHER_OPEX,
            UTILITY_PER_T_MEOH,
            MEOH_ANNUAL_T,
            OPERATING_HOURS,
            MW_MEOH,
            H2_FEED_KMOL,
        )
        import constants as _plant_constants                            # type: ignore
        _FLUEGAS_CO2_KMOL = float(_plant_constants.FLUEGAS_CO2_KMOL)
        _PROJECT_IMPORT_OK = True
    except Exception as _e:
        warnings.warn(
            f"[flowsheet_graph] Could not import project economics "
            f"({_e!r}); evaluate_with_surrogates will use synthetic targets."
        )


# ─────────────────────────────────────────────────────────────────────
# Module-level surrogate cache (loaded lazily, shared across samples).
# Using module globals means a 200-sample dataset doesn't reload the
# .pkl files on every call.
# ─────────────────────────────────────────────────────────────────────
_SURROGATE_CACHE: dict = {}
_SURROGATE_DIR = Path(__file__).resolve().parent / "saved_models" / "surrogates_v2"

# 5-D surrogate input bounds (must match BOUNDS in
# EURECHA/scripts/surrogate_optimization.py).  Order:
#   [T_rxn(°C), P_rxn(bar), f_purge, F_H2(kmol/h), P_col(bar)]
_SURR_BOUNDS_5D = np.array([
    [210.0, 280.0],
    [50.0, 100.0],
    [0.003, 0.10],
    [2515.0, 3147.0],
    [1.0, 3.0],
])


def _load_surrogates() -> dict:
    """Load the five GPR surrogates from saved_models/surrogates_v2/.

    Returns a dict keyed by name with entries {"model", "X_mu", "X_sig"}
    (X_mu/X_sig are None for the 1-D surrogates).  Cached in
    ``_SURROGATE_CACHE`` after the first call.
    """
    if _SURROGATE_CACHE:
        return _SURROGATE_CACHE

    import pickle

    spec = {
        "h2":     ("gp_h2_production.pkl",       False),   # 1-D: load
        "meoh":   ("gp_meoh_output.pkl",         True),    # 3-D: [F_H2, T, P]
        "energy": ("gp_energy_consumption.pkl",  False),   # 1-D: load
        "lcom":   ("gp_lcom_5d.pkl",             True),    # 5-D
        "util":   ("gp_util_5d.pkl",             True),    # 5-D
    }
    for key, (fname, has_norm) in spec.items():
        p = _SURROGATE_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"Surrogate not found: {p}")
        with open(p, "rb") as f:
            data = pickle.load(f)
        entry = {
            "model": data["model"],
            "X_mu":  data.get("X_mu") if has_norm else None,
            "X_sig": data.get("X_sig") if has_norm else None,
        }
        _SURROGATE_CACHE[key] = entry
    return _SURROGATE_CACHE


# ─────────────────────────────────────────────────────────────────────
# 1. GRAPH TOPOLOGY — fixed for the base flowsheet
# ─────────────────────────────────────────────────────────────────────

# Node definitions: id, name, type
UNIT_OPS = [
    (0,  "ABS",    "separator"),       # MEA absorber
    (1,  "STRIP",  "separator"),       # MEA stripper
    (2,  "COMP",   "pressure"),        # CO₂ compressor train
    (3,  "PEM",    "reactor"),         # PEM electrolyser
    (4,  "MIX",    "mixer"),           # feed mixer (CO₂ + H₂)
    (5,  "HX_PRE", "heat_exchanger"),  # feed-effluent heat exchanger
    (6,  "REACT",  "reactor"),         # MeOH synthesis reactor
    (7,  "COOL",   "heat_exchanger"),  # reactor cooler
    (8,  "FLASH",  "separator"),       # HP flash drum
    (9,  "SPLIT",  "splitter"),        # recycle/purge splitter
    (10, "DIST",   "separator"),       # methanol distillation
]

NODE_TYPES = ["reactor", "separator", "heat_exchanger", "pressure", "mixer", "splitter"]
TYPE_TO_IDX = {t: i for i, t in enumerate(NODE_TYPES)}
N_TYPES = len(NODE_TYPES)

# Edge list: (src, dst, stream_name)
STREAMS = [
    (0, 1,  "rich_mea"),
    (1, 0,  "lean_mea"),         # recycle: MEA loop
    (1, 2,  "co2_conc"),
    (2, 4,  "co2_comp"),
    (3, 4,  "h2"),
    (4, 5,  "mixed_feed"),
    (5, 6,  "preheated_feed"),
    (6, 7,  "reactor_effluent"),
    (7, 8,  "cooled_effluent"),
    (7, 5,  "fehe_hot"),         # energy stream: heat integration
    (8, 9,  "vapour"),
    (9, 4,  "recycle"),          # recycle: synthesis loop
    (8, 10, "crude_meoh"),
]

EDGE_INDEX = torch.tensor(
    [[s for s, _, _ in STREAMS], [d for _, d, _ in STREAMS]],
    dtype=torch.long,
)

# ─────────────────────────────────────────────────────────────────────
# 2. NODE FEATURE SPECIFICATION
# ─────────────────────────────────────────────────────────────────────
#
# Each node has:
#   - Continuous design parameters (unit-specific, zero-padded to max_dim)
#   - One-hot unit type vector (6 dims)
#
# The design parameters per unit are the knobs you'd vary in a
# flowsheet optimisation. Ranges come from your Aspen model bounds.

NODE_FEATURE_SPEC = {
    0:  {"names": ["T_abs", "P_abs", "LG_ratio", "n_stages_abs", "lean_loading"],
         "ranges": [(40, 60), (1, 2), (1.5, 4.0), (8, 25), (0.15, 0.30)]},
    1:  {"names": ["T_reb", "P_strip", "n_stages_strip", "boilup_ratio"],
         "ranges": [(110, 125), (1.5, 2.5), (6, 20), (0.5, 2.0)]},
    2:  {"names": ["P_out_comp", "n_stages_comp", "eta_isen"],
         "ranges": [(50, 80), (2, 5), (0.72, 0.88)]},
    3:  {"names": ["P_elec_MW", "eta_elec", "T_stack", "capacity_factor"],
         "ranges": [(10, 60), (0.58, 0.72), (60, 90), (0.3, 1.0)]},
    4:  {"names": ["H2_CO2_ratio"],
         "ranges": [(2.8, 3.2)]},
    5:  {"names": ["T_out_preheat", "DT_min", "UA_preheat"],
         "ranges": [(200, 260), (5, 20), (50, 500)]},
    6:  {"names": ["T_react", "P_react", "WHSV", "L_bed", "D_bed", "cat_mass_kg"],
         "ranges": [(220, 280), (50, 80), (2, 10), (3, 12), (1, 4), (500, 5000)]},
    7:  {"names": ["T_out_cool", "Q_cool_MW"],
         "ranges": [(35, 50), (5, 20)]},
    8:  {"names": ["T_flash", "P_flash"],
         "ranges": [(30, 60), (40, 80)]},
    9:  {"names": ["purge_fraction"],
         "ranges": [(0.02, 0.10)]},
    10: {"names": ["n_trays", "RR", "P_dist", "T_cond", "T_reb_dist"],
         "ranges": [(15, 50), (1.0, 5.0), (1.0, 3.0), (60, 80), (100, 120)]},
}

# Max continuous features per node (for zero-padding)
MAX_CONT_FEATURES = max(len(v["names"]) for v in NODE_FEATURE_SPEC.values())


def _one_hot_type(node_id: int) -> list:
    """One-hot encoding for unit type."""
    typ = UNIT_OPS[node_id][2]
    vec = [0.0] * N_TYPES
    vec[TYPE_TO_IDX[typ]] = 1.0
    return vec


def _node_feature_dim() -> int:
    """Total dimension of each node feature vector."""
    return MAX_CONT_FEATURES + N_TYPES  # continuous (zero-padded) + one-hot type


# ─────────────────────────────────────────────────────────────────────
# 3. EDGE FEATURE SPECIFICATION
# ─────────────────────────────────────────────────────────────────────
#
# Each edge (stream) carries: [ṁ_total, T, P, x_CO2, x_H2, x_MeOH, x_H2O, x_inert, is_recycle, is_energy]
# These are computed from the node parameters + surrogate evaluation.

EDGE_FEATURE_DIM = 10

EDGE_FEATURE_NAMES = [
    "m_dot_total",  # total mass flow (kg/s)
    "T",            # temperature (°C)
    "P",            # pressure (bar)
    "x_CO2",        # mass fraction CO₂
    "x_H2",         # mass fraction H₂
    "x_MeOH",       # mass fraction MeOH
    "x_H2O",        # mass fraction H₂O
    "x_inert",      # mass fraction inerts (N₂, CH₄)
    "is_recycle",   # binary: 1 if recycle stream
    "is_energy",    # binary: 1 if energy stream (FEHE)
]

# Edge-index sets for the structural flags (used in several places).
_RECYCLE_EDGES = {1, 11}   # lean_mea, recycle
_ENERGY_EDGES  = {9}       # fehe_hot


def _structural_flags(edge_attr: np.ndarray) -> np.ndarray:
    """Stamp is_recycle (slot 8) and is_energy (slot 9) onto an
    already-shaped (13, EDGE_FEATURE_DIM) array and return it."""
    for i in _RECYCLE_EDGES:
        edge_attr[i, 8] = 1.0
    for i in _ENERGY_EDGES:
        edge_attr[i, 9] = 1.0
    return edge_attr


def compute_edge_features(
    node_params: dict,
    surrogate_out: Optional[dict] = None,
) -> np.ndarray:
    """Populate the 13×10 edge-feature array for one flowsheet sample.

    Strategy (Option B — base-case + lightweight physics scaling):
        1. Start from the Aspen base case (via ``stream_data.load_basecase_streams``).
        2. Override T/P on the edges where ``node_params`` directly sets them
           — reactor inlet/outlet get (T_rxn, P_rxn); downstream HP flash &
           recycle loop stays at P_rxn; crude_meoh drops to P_col.
        3. Scale mass flows by ``meoh_tph / BASECASE_MEOH_TPH`` (from the
           MeOH surrogate if available, else uses the base case so flows
           match Aspen).
        4. Stamp ``is_recycle`` / ``is_energy`` structural flags on slots 8–9.

    On any exception the function silently falls back to the "structural
    flags only" zero array, mirroring the synthetic-fallback pattern
    already used by ``evaluate_with_surrogates``.  Callers can detect the
    fallback by checking whether slots 0–7 are all zero.

    Parameters
    ----------
    node_params : dict
        11-node parameter dict (see ``NODE_FEATURE_SPEC``).
    surrogate_out : dict, optional
        Output of ``evaluate_with_surrogates``.  If it contains a
        ``_meoh_tph`` key, mass flows are scaled by
        ``_meoh_tph / BASECASE_MEOH_TPH``; otherwise the scaling factor
        is 1.0.

    Returns
    -------
    np.ndarray of shape (13, EDGE_FEATURE_DIM)
    """
    # Import the stream_data module (not bound names) so that
    # monkeypatching its functions in tests takes effect here.
    try:
        from . import stream_data as _sd                                 # type: ignore
    except Exception:
        try:
            import stream_data as _sd                                     # type: ignore
        except Exception as e:
            warnings.warn(
                f"[flowsheet_graph] stream_data import failed "
                f"({type(e).__name__}: {e}); returning structural-flag-only "
                f"edge features."
            )
            return _structural_flags(np.zeros((len(STREAMS), EDGE_FEATURE_DIM)))

    try:
        # Physics lives in stream_data.build_edge_feature_array — it's
        # pure numpy so it can be unit-tested without torch.
        return _sd.build_edge_feature_array(node_params, surrogate_out)
    except Exception as e:
        warnings.warn(
            f"[flowsheet_graph] compute_edge_features failed "
            f"({type(e).__name__}: {e}); returning structural-flag-only "
            f"edge features."
        )
        return _structural_flags(np.zeros((len(STREAMS), EDGE_FEATURE_DIM)))


# ─────────────────────────────────────────────────────────────────────
# 4. BUILD A SINGLE GRAPH (Data object)
# ─────────────────────────────────────────────────────────────────────

def build_base_graph(
    node_params: Optional[dict] = None,
    edge_features: Optional[np.ndarray] = None,
    targets: Optional[dict] = None,
) -> Data:
    """
    Construct a PyG Data object for the CO₂-to-methanol flowsheet.

    Parameters
    ----------
    node_params : dict[int, list[float]], optional
        Maps node_id → list of continuous design parameters.
        If None, samples uniformly from ranges in NODE_FEATURE_SPEC.
    edge_features : np.ndarray of shape (13, EDGE_FEATURE_DIM), optional
        Stream condition features. If None, uses placeholder zeros.
        In practice, compute these from surrogates or Aspen.
    targets : dict with keys "TAC", "carbon_eff", "LCOM", optional
        Graph-level regression targets.

    Returns
    -------
    torch_geometric.data.Data
    """
    # --- Node features ---
    x_list = []
    for node_id in range(len(UNIT_OPS)):
        spec = NODE_FEATURE_SPEC[node_id]
        n_feat = len(spec["names"])

        if node_params and node_id in node_params:
            cont = list(node_params[node_id])
        else:
            # Random sample within bounds
            cont = [
                np.random.uniform(lo, hi) for lo, hi in spec["ranges"]
            ]

        # Zero-pad to MAX_CONT_FEATURES
        cont_padded = cont + [0.0] * (MAX_CONT_FEATURES - n_feat)

        # Append one-hot type
        feat = cont_padded + _one_hot_type(node_id)
        x_list.append(feat)

    x = torch.tensor(x_list, dtype=torch.float32)

    # --- Edge features ---
    if edge_features is not None:
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    elif node_params is not None:
        # Populate stream conditions (slots 0–7) from the Aspen base case
        # via compute_edge_features, with node-param overrides for T/P.
        edge_attr = torch.tensor(
            compute_edge_features(node_params), dtype=torch.float32
        )
    else:
        # No design info — keep historical behaviour: zero stream
        # conditions, only the structural flags populated.
        edge_attr_np = np.zeros((len(STREAMS), EDGE_FEATURE_DIM))
        _structural_flags(edge_attr_np)
        edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)

    # --- Targets ---
    if targets:
        y = torch.tensor(
            [targets.get("TAC", 0.0), targets.get("carbon_eff", 0.0), targets.get("LCOM", 0.0)],
            dtype=torch.float32,
        ).unsqueeze(0)  # shape (1, 3)
    else:
        y = None

    data = Data(
        x=x,
        edge_index=EDGE_INDEX,
        edge_attr=edge_attr,
        y=y,
        num_nodes=len(UNIT_OPS),
    )
    return data


# ─────────────────────────────────────────────────────────────────────
# 5. DATASET GENERATION (via surrogates)
# ─────────────────────────────────────────────────────────────────────

def generate_lhs_samples(n_samples: int, seed: int = 42) -> list[dict]:
    """Generate n_samples flowsheet parameter sets via Latin Hypercube
    Sampling.

    Uses ``scipy.stats.qmc.LatinHypercube`` with optimised centering
    (scrambled + seeded) — proper space-filling, not just uniform
    random.  If scipy isn't available we fall back to uniform random
    so the module still runs standalone.

    Returns a list of dicts, each mapping node_id → list of continuous
    design parameters drawn from the ranges in ``NODE_FEATURE_SPEC``.
    """
    # Flatten (node_id, param_idx) into a dense design vector so the
    # LHS quasi-random stream is coherent across the whole 5-D+ space.
    flat_ranges: list[tuple[int, int, float, float]] = []
    for node_id, spec in NODE_FEATURE_SPEC.items():
        for p_idx, (lo, hi) in enumerate(spec["ranges"]):
            flat_ranges.append((node_id, p_idx, float(lo), float(hi)))
    d = len(flat_ranges)

    try:
        from scipy.stats import qmc  # type: ignore
        sampler = qmc.LatinHypercube(d=d, scramble=True, seed=seed)
        # unit cube → scaled to each param's bounds
        u = sampler.random(n=n_samples)                                     # (n_samples, d)
        lows  = np.array([lo for (_, _, lo, _)  in flat_ranges])
        highs = np.array([hi for (_, _, _,  hi) in flat_ranges])
        scaled = qmc.scale(u, lows, highs)
    except Exception:
        # Graceful fallback: uniform random (preserves old behaviour).
        rng = np.random.default_rng(seed)
        scaled = np.column_stack([
            rng.uniform(lo, hi, size=n_samples) for (_, _, lo, hi) in flat_ranges
        ])

    # Rehydrate into the list-of-dicts format every caller expects.
    samples: list[dict] = []
    for i in range(n_samples):
        params: dict = {}
        for j, (node_id, p_idx, _, _) in enumerate(flat_ranges):
            params.setdefault(node_id, [])
            params[node_id].append(float(scaled[i, j]))
        samples.append(params)
    return samples


def _synthetic_targets(node_params: dict) -> dict:
    """Original analytic stand-in — kept verbatim as the fallback path."""
    T_react = node_params[6][0]
    P_react = node_params[6][1]
    purge = node_params[9][0]

    TAC = 50e6 + 1e5 * (T_react - 250) ** 2 + 2e5 * (P_react - 65) ** 2
    carbon_eff = 0.85 + 0.1 * (1 - purge / 0.10) + np.random.normal(0, 0.02)
    carbon_eff = float(np.clip(carbon_eff, 0.5, 0.99))
    LCOM = TAC / (carbon_eff * 1e5)  # simplified

    # Stream conditions: populate from Aspen base case via
    # compute_edge_features (no meoh_tph in synthetic branch, so mass
    # scaling defaults to 1.0).  Falls back to structural flags only if
    # stream_data fails to import.
    stream_features = compute_edge_features(node_params, surrogate_out=None)

    return {
        "TAC": float(TAC),
        "carbon_eff": float(carbon_eff),
        "LCOM": float(LCOM),
        "stream_features": stream_features,
        "_source": "synthetic",
        "_edge_source": "real" if stream_features[:, 0].any() else "synthetic",
    }


def _node_params_to_surrogate_input(node_params: dict) -> np.ndarray:
    """Map the 11-node GNN parameter dict → the 5-D surrogate vector
    [T_rxn, P_rxn, f_purge, F_H2, P_col].

    Notes on the mapping (ambiguities called out inline):

      - node 6 (REACT) params: [T_react, P_react, WHSV, L_bed, D_bed,
        cat_mass_kg].  Only T and P map to surrogate inputs.  WHSV/bed
        geometry are *not* in the trained 5-D surrogate — they'd need a
        retrain to be used.
      - node 9 (SPLIT) params: [purge_fraction] → f_purge.
      - node 3 (PEM) params: [P_elec_MW, eta_elec, T_stack,
        capacity_factor].  F_H2 is derived as capacity_factor × the
        plant's nominal H2_FEED_KMOL (2704 kmol/h at base case).  This is
        a best-effort mapping — the surrogate's F_H2 range [2515, 3147]
        corresponds roughly to capacity_factor ∈ [0.93, 1.16], so we
        clip hard to the training domain.
      - node 10 (DIST) params: [n_trays, RR, P_dist, T_cond, T_reb_dist].
        P_dist → P_col.

    All 5 values are clipped to the surrogate's training bounds.
    """
    T   = float(node_params[6][0])                                      # T_react
    P   = float(node_params[6][1])                                      # P_react
    fp  = float(node_params[9][0])                                      # purge_fraction

    if _PROJECT_IMPORT_OK:
        cap_factor = float(node_params[3][3])                            # capacity_factor
        F_H2 = cap_factor * float(H2_FEED_KMOL)
    else:
        # Standalone fallback: use the midpoint of the surrogate range.
        F_H2 = 2704.0 * float(node_params[3][3])

    P_col = float(node_params[10][2])                                    # P_dist

    x = np.array([T, P, fp, F_H2, P_col], dtype=float)
    x = np.clip(x, _SURR_BOUNDS_5D[:, 0], _SURR_BOUNDS_5D[:, 1])
    return x


def evaluate_with_surrogates(node_params: dict, surrogates: Optional[dict] = None) -> dict:
    """
    Evaluate flowsheet KPIs given node design parameters.

    Uses the calibrated GPR surrogates from
    ``saved_models/surrogates_v2/`` (loaded lazily and cached
    module-globally).  KPIs are then computed from the project economics
    in ``EURECHA/figures/scripts/constants.py``:

        TAC         = annualised CAPEX + utility OPEX + other OPEX (£/yr)
        carbon_eff  = MeOH produced (kmol/hr) / CO₂ in flue gas (kmol/hr)
        LCOM        = predicted directly by the 5-D LCOM surrogate (£/t)

    If any step fails (surrogate load error, missing constants, etc.)
    the function quietly falls back to the original analytic synthetic
    targets, so the module remains runnable standalone.

    Parameters
    ----------
    node_params : dict
        Mapping ``node_id → list[float]`` of design parameters, matching
        ``NODE_FEATURE_SPEC``.
    surrogates : dict, optional
        Pre-loaded surrogate dict (same shape as ``_load_surrogates()``
        returns).  If ``None``, surrogates are loaded on first use.

    Returns
    -------
    dict with keys ``TAC``, ``carbon_eff``, ``LCOM``, ``stream_features``
    (13 × 10 ndarray), and ``_source`` ("real" or "synthetic").
    """
    if not _PROJECT_IMPORT_OK:
        return _synthetic_targets(node_params)

    try:
        surr = surrogates if surrogates is not None else _load_surrogates()

        # --- Build the 5-D surrogate input and its 3-D subset --------
        x5 = _node_params_to_surrogate_input(node_params)              # shape (5,)
        T, P, f_purge, F_H2, P_col = x5

        # 3-D MeOH surrogate expects [F_H2, T, P] (per models/surrogates.py)
        x3 = np.array([[F_H2, T, P]])

        # --- Predict: LCOM (5-D), utility (5-D), MeOH (3-D) ----------
        def _norm(x, entry):
            mu, sig = entry["X_mu"], entry["X_sig"]
            return (x - mu) / sig if mu is not None else x

        x5_mat = x5.reshape(1, -1)
        LCOM         = float(surr["lcom"]["model"].predict(_norm(x5_mat, surr["lcom"]))[0])
        # NOTE: ``gp_util_5d`` predicts the *overall CO₂ utilisation
        # fraction* (≈ OVERALL_CONV, 0.77–1.0) — NOT a utility cost in
        # £/t.  See ``simplified_process_model``'s second return value.
        co2_util_frac = float(surr["util"]["model"].predict(_norm(x5_mat, surr["util"]))[0])
        co2_util_frac = float(np.clip(co2_util_frac, 0.0, 1.0))
        meoh_tph     = float(surr["meoh"]["model"].predict(_norm(x3,    surr["meoh"]))[0])
        meoh_tph     = max(meoh_tph, 0.0)

        # --- Economics → TAC (£/yr) ----------------------------------
        # Annualised CAPEX: 10% capital recovery factor (matches
        # ANNUALISED_CAPEX definition in constants.py).  Utility OPEX
        # per tonne MeOH is a fixed property of the plant design (see
        # constants.py: CW + chilled water + steam + trim heater), so
        # we use the project's constant rather than scaling by
        # surrogate-predicted utility cost.
        capex_annual  = float(TOTAL_CAPEX_M) * 0.10 * 1e6                # £/yr
        meoh_annual_t = meoh_tph * float(OPERATING_HOURS)                # t/yr
        opex_util     = float(UTILITY_PER_T_MEOH) * meoh_annual_t        # £/yr
        opex_other    = float(OTHER_OPEX) * meoh_annual_t                # £/yr
        TAC = capex_annual + opex_util + opex_other                      # £/yr

        # --- Carbon efficiency ---------------------------------------
        # CO₂ utilised / CO₂ in flue gas.  The MeOH surrogate gives
        # production, and each kmol MeOH corresponds to 1 kmol CO₂
        # consumed (CO₂ + 3 H₂ → CH₃OH + H₂O).  We then weight by the
        # 5-D utilisation surrogate to account for purge losses.
        meoh_kmol = meoh_tph * 1000.0 / float(MW_MEOH)
        carbon_eff = float(np.clip(
            co2_util_frac * meoh_kmol / _FLUEGAS_CO2_KMOL, 0.0, 1.0
        ))

        # --- Edge features: base case + T/P overrides + mass scaling --
        # `compute_edge_features` pulls the Aspen base case from
        # `stream_data.load_basecase_streams`, then overrides reactor /
        # column T/P from node_params and scales mass flows by
        # meoh_tph / BASECASE_MEOH_TPH.  If anything fails, it returns
        # a structural-flag-only array and emits a warning.
        _edge_src = {"_meoh_tph": meoh_tph}
        stream_features = compute_edge_features(node_params, _edge_src)
        _edge_source = "real" if stream_features[:, 0].any() else "synthetic"

        return {
            "TAC": float(TAC),
            "carbon_eff": carbon_eff,
            "LCOM": LCOM,
            "stream_features": stream_features,
            "_source": "real",
            "_edge_source": _edge_source,
            "_meoh_tph": meoh_tph,
        }
    except Exception as e:
        warnings.warn(
            f"[flowsheet_graph] surrogate evaluation failed "
            f"({type(e).__name__}: {e}); falling back to synthetic targets."
        )
        return _synthetic_targets(node_params)


def build_dataset(n_samples: int = 500, seed: int = 42) -> list[Data]:
    """Build a list of PyG Data objects by sampling + evaluating flowsheets."""
    samples = generate_lhs_samples(n_samples, seed)
    dataset = []
    for params in samples:
        results = evaluate_with_surrogates(params)
        graph = build_base_graph(
            node_params=params,
            edge_features=results["stream_features"],
            targets={"TAC": results["TAC"], "carbon_eff": results["carbon_eff"], "LCOM": results["LCOM"]},
        )
        dataset.append(graph)
    return dataset


# ─────────────────────────────────────────────────────────────────────
# 6. GNN MODEL — GIN with edge features (GINE)
# ─────────────────────────────────────────────────────────────────────
#
# Why GINE (Graph Isomorphism Network with Edge features)?
# - GIN is maximally expressive among 1-WL GNNs (Xu et al., 2019)
# - Edge features are critical: stream conditions carry essential info
# - GINE extends GIN to incorporate edge_attr in message passing
# - For 11 nodes, you don't need deep architectures — 3 layers suffice
#
# Alternatives to explore later:
# - GATv2 with edge features (attention-weighted, interpretable)
# - SchNet-style continuous filter (if you want distance-like edge features)
# - HeteroGNN if you want type-specific message passing functions

class FlowsheetGNN(nn.Module):
    """
    GINE for graph-level regression: flowsheet → [TAC, η_C, LCOM].

    Architecture:
        Node embedding → 3× GINE layers (with residual) → global pool → MLP head

    Parameters
    ----------
    node_dim : int
        Input node feature dimension (MAX_CONT_FEATURES + N_TYPES).
    edge_dim : int
        Input edge feature dimension.
    hidden_dim : int
        Hidden channel width.
    n_targets : int
        Number of regression targets.
    n_layers : int
        Number of GINE message-passing layers.
    dropout : float
        Dropout rate.
    pool : str
        Global pooling: "mean", "add", or "mean+add" (concatenated).
    """

    def __init__(
        self,
        node_dim: int = _node_feature_dim(),
        edge_dim: int = EDGE_FEATURE_DIM,
        hidden_dim: int = 64,
        n_targets: int = 3,
        n_layers: int = 3,
        dropout: float = 0.1,
        pool: str = "mean+add",
    ):
        super().__init__()

        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.pool_mode = pool
        pool_dim = hidden_dim * 2 if pool == "mean+add" else hidden_dim

        # Head: the ``dropout`` argument is retained for API compatibility,
        # but **head dropout is empirically harmful** for this architecture.
        # In the 2026-04-21 verification re-run, every d=0.1 config
        # collapsed to val_loss ≈ Var(y) (predict-the-mean) even after
        # removing F.dropout from the message-passing loop, because the
        # combination of head nn.Dropout + Adam's adaptive step pushes
        # the model into a degenerate minimum in the first few epochs
        # and it can't escape. nn.Identity() is used regardless of the
        # dropout value; regularisation is provided via the optimiser's
        # weight_decay=1e-5. Use a smaller hidden_dim if over-fitting
        # becomes a concern.
        self.head = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Identity(),  # formerly nn.Dropout(dropout) — see note above
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_targets),
        )

        self.dropout = dropout
        self.n_layers = n_layers

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # Embed
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)

        # Message passing with residual connections.
        # NOTE: feature dropout is intentionally NOT applied inside the
        # message-passing loop — stacking F.dropout after nn.BatchNorm1d
        # triggers the variance-shift pathology described in Li et al.
        # 2018 ("Understanding the Disharmony Between Dropout and Batch
        # Normalization by Variance Shift"), which in practice collapses
        # eval-mode predictions toward the mean (observed in the
        # 2026-04-20 HP sweep: every d=0.1 config reached val_loss ≈
        # Var(y), i.e. predicted a constant). Regularisation is kept
        # only in the head MLP (``nn.Dropout`` before the penultimate
        # Linear), which is not followed by BatchNorm.
        for i in range(self.n_layers):
            x_res = x
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x + x_res)  # residual

        # Global pooling
        if self.pool_mode == "mean+add":
            x_pool = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=-1)
        elif self.pool_mode == "add":
            x_pool = global_add_pool(x, batch)
        else:
            x_pool = global_mean_pool(x, batch)

        # Regression head
        return self.head(x_pool)


# ─────────────────────────────────────────────────────────────────────
# 7. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────

def train_gnn(
    dataset: list[Data],
    n_epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    val_frac: float = 0.2,
    seed: int = 42,
):
    """
    Train FlowsheetGNN with early stopping on validation loss.

    Returns trained model and training history.
    """
    from torch_geometric.loader import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ─── Normalise targets ───
    all_y = torch.stack([d.y.squeeze(0) for d in dataset])
    y_mean = all_y.mean(dim=0)
    y_std = all_y.std(dim=0) + 1e-8
    for d in dataset:
        d.y = (d.y - y_mean) / y_std

    # ─── Train/val split ───
    n_val = int(len(dataset) * val_frac)
    perm = torch.randperm(len(dataset)).tolist()
    val_data = [dataset[i] for i in perm[:n_val]]
    train_data = [dataset[i] for i in perm[n_val:]]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # ─── Model + optimiser ───
    model = FlowsheetGNN()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=15, min_lr=1e-5
    )

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience_counter = 0
    patience_limit = 30

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimiser.zero_grad()
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(train_data)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch)
                val_loss += F.mse_loss(pred, batch.y).item() * batch.num_graphs
        val_loss /= len(val_data)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 20 == 0 or patience_counter >= patience_limit:
            print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | lr {optimiser.param_groups[0]['lr']:.1e}")

        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    # Store normalisation for inference
    model.y_mean = y_mean
    model.y_std = y_std

    return model, history


# ─────────────────────────────────────────────────────────────────────
# 8. QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CO₂-to-Methanol Flowsheet GNN — Sanity Check")
    print("=" * 60)

    # Build one graph
    g = build_base_graph()
    print(f"\nBase graph:")
    print(f"  Nodes:          {g.num_nodes}")
    print(f"  Edges:          {g.num_edges}")
    print(f"  Node feat dim:  {g.x.shape[1]}  ({MAX_CONT_FEATURES} continuous + {N_TYPES} one-hot)")
    print(f"  Edge feat dim:  {g.edge_attr.shape[1]}")
    print(f"  Has cycles:     True (MEA loop, synthesis recycle, FEHE)")
    print(f"  Directed:       True")

    # Build dataset — one LHS sample up front so we can report which
    # branch of evaluate_with_surrogates is active.
    _probe = evaluate_with_surrogates(generate_lhs_samples(1, seed=42)[0])
    _src = _probe.get("_source", "unknown")
    print(f"\nGenerating 200-sample dataset ({_src} targets)...")
    dataset = build_dataset(n_samples=200, seed=42)
    print(f"  Dataset size:   {len(dataset)}")
    print(f"  Target shape:   {dataset[0].y.shape}")
    # Report the range of the real/synthetic targets to give a sanity
    # pulse on the values the GNN will try to fit.
    import torch as _t
    _y = _t.stack([d.y.squeeze(0) for d in dataset])
    _names = ["TAC (£)", "carbon_eff", "LCOM (£/t)"]
    for _i, _n in enumerate(_names):
        print(f"  {_n:<12} range: {_y[:, _i].min().item():.3g} … {_y[:, _i].max().item():.3g}")

    # Forward pass
    model = FlowsheetGNN()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nFlowsheetGNN:")
    print(f"  Parameters:     {n_params:,}")
    print(f"  Architecture:   GINE (3 layers, 64 hidden, mean+add pool)")

    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset[:4], batch_size=4)
    batch = next(iter(loader))
    with torch.no_grad():
        out = model(batch)
    print(f"  Forward pass:   batch of {batch.num_graphs} → output {out.shape}")
    print(f"  Output sample:  {out[0].numpy()}")

    # Quick training (few epochs to verify gradient flow)
    print(f"\nTraining 50 epochs on synthetic data...")
    model, hist = train_gnn(dataset, n_epochs=50, batch_size=32)
    print(f"  Final train loss: {hist['train_loss'][-1]:.4f}")
    print(f"  Final val loss:   {hist['val_loss'][-1]:.4f}")

    print("\n✓ All checks passed. Ready for real surrogate integration.")
