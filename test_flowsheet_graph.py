"""
Unit tests for flowsheet_graph.py
=================================
Pytest-compatible smoke tests covering:

  1. Graph dimensions (11 nodes, 13 edges, node_dim=12, edge_dim=10)
  2. Varied targets under random sampling (no identical duplicates)
  3. Forward pass + gradient flow (loss.backward() does not error)
  4. Target normalisation is reversible (denormalised predictions land
     in a physically reasonable range)
  5. Cycle structure (MEA loop, synthesis recycle, FEHE)

Run:
    cd EURECHA/rl_dynamic_control/
    pytest -q test_flowsheet_graph.py
    # or
    python test_flowsheet_graph.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from flowsheet_graph import (
    EDGE_FEATURE_DIM,
    MAX_CONT_FEATURES,
    N_TYPES,
    STREAMS,
    UNIT_OPS,
    NODE_FEATURE_SPEC,
    FlowsheetGNN,
    build_base_graph,
    build_dataset,
    compute_edge_features,
    evaluate_with_surrogates,
    generate_lhs_samples,
    _node_feature_dim,
)


# ─────────────────────────────────────────────────────────────────────
# 1. Graph dimensions
# ─────────────────────────────────────────────────────────────────────

def test_graph_shapes_match_spec() -> None:
    """build_base_graph() produces the expected shapes."""
    g = build_base_graph()

    # 11 unit operations, 13 directed streams
    assert g.num_nodes == 11 == len(UNIT_OPS)
    assert g.num_edges == 13 == len(STREAMS)

    # x : (num_nodes, node_dim)
    assert g.x.shape == (11, _node_feature_dim())
    assert g.x.shape[1] == MAX_CONT_FEATURES + N_TYPES == 12
    # 6 continuous + 6 one-hot = 12

    # edge_index : (2, num_edges)
    assert g.edge_index.shape == (2, 13)

    # edge_attr : (num_edges, edge_dim)
    assert g.edge_attr.shape == (13, EDGE_FEATURE_DIM) == (13, 10)


def test_one_hot_type_is_valid() -> None:
    """Each node has exactly one type bit set (one-hot constraint)."""
    g = build_base_graph()
    type_block = g.x[:, MAX_CONT_FEATURES:]  # shape (11, N_TYPES)
    assert type_block.shape == (11, N_TYPES)
    # Each row should sum to 1.0 and have a single maximum.
    row_sums = type_block.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(11))
    assert (type_block.max(dim=1).values == 1.0).all()


# ─────────────────────────────────────────────────────────────────────
# 2. Cycle structure
# ─────────────────────────────────────────────────────────────────────

def test_cycles_present_in_edge_index() -> None:
    """The three documented cycles exist in the edge index."""
    g = build_base_graph()
    edges = {(int(s), int(d)) for s, d in g.edge_index.t().tolist()}
    # MEA loop: 0 ↔ 1
    assert (0, 1) in edges and (1, 0) in edges
    # Synthesis recycle: 9 → 4 closes the loop
    assert (9, 4) in edges
    # FEHE energy stream: 7 → 5
    assert (7, 5) in edges


# ─────────────────────────────────────────────────────────────────────
# 3. Target variance under random sampling
# ─────────────────────────────────────────────────────────────────────

def test_targets_are_varied_across_samples() -> None:
    """Random designs should produce a meaningful spread of KPIs —
    otherwise the GNN has nothing to learn."""
    samples = generate_lhs_samples(n_samples=32, seed=0)
    results = [evaluate_with_surrogates(p) for p in samples]

    tac = np.array([r["TAC"] for r in results])
    ceff = np.array([r["carbon_eff"] for r in results])
    lcom = np.array([r["LCOM"] for r in results])

    # Not all identical
    assert tac.std() > 0.0
    assert ceff.std() > 0.0
    assert lcom.std() > 0.0

    # And physically plausible (generous bounds so either branch passes)
    assert tac.min() > 1e6 and tac.max() < 1e10          # £/yr
    assert 0.0 <= ceff.min() and ceff.max() <= 1.0
    assert 100.0 < lcom.min() and lcom.max() < 5000.0    # £/t MeOH


# ─────────────────────────────────────────────────────────────────────
# 4. Forward + backward pass
# ─────────────────────────────────────────────────────────────────────

def test_forward_backward_gradient_flow() -> None:
    """Gradients flow through every module — no NaNs, no disconnected
    parameters."""
    from torch_geometric.loader import DataLoader

    dataset = build_dataset(n_samples=8, seed=0)
    # Normalise targets so the loss doesn't explode on the first batch.
    all_y = torch.stack([d.y.squeeze(0) for d in dataset])
    mu, sig = all_y.mean(0), all_y.std(0) + 1e-8
    for d in dataset:
        d.y = (d.y - mu) / sig

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = FlowsheetGNN()
    model.train()

    batch = next(iter(loader))
    pred = model(batch)
    loss = F.mse_loss(pred, batch.y)

    assert torch.isfinite(loss), "loss is NaN / inf before backward"
    loss.backward()

    # Every trainable parameter should receive a gradient.
    missing = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not missing, f"Params missing gradients: {missing}"

    # And no grad should contain NaNs.
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN grad in {name}"


def test_model_parameter_count_reasonable() -> None:
    """Sanity: 3-layer, 64-hidden GINE should be ~30k–200k params."""
    model = FlowsheetGNN()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 20_000 < n_params < 250_000, f"Unexpected model size: {n_params:,}"


# ─────────────────────────────────────────────────────────────────────
# 5. Normalisation reversibility
# ─────────────────────────────────────────────────────────────────────

def test_target_normalisation_is_reversible() -> None:
    """Normalising targets and denormalising back recovers physical
    ranges (within a tolerance)."""
    dataset = build_dataset(n_samples=16, seed=1)
    y_raw = torch.stack([d.y.squeeze(0) for d in dataset])    # (N, 3)
    mu = y_raw.mean(0)
    sig = y_raw.std(0) + 1e-8

    # Normalise then denormalise
    y_norm = (y_raw - mu) / sig
    y_back = y_norm * sig + mu

    assert torch.allclose(y_back, y_raw, atol=1e-3)

    # Denormalised values land in physically reasonable ranges
    tac_back = y_back[:, 0]
    ceff_back = y_back[:, 1]
    lcom_back = y_back[:, 2]
    assert (tac_back > 1e6).all() and (tac_back < 1e10).all()
    assert (ceff_back >= 0.0).all() and (ceff_back <= 1.0).all()
    assert (lcom_back > 100.0).all() and (lcom_back < 5000.0).all()


# ─────────────────────────────────────────────────────────────────────
# 6. Edge features — stream conditions on the 13 edges (Option B)
# ─────────────────────────────────────────────────────────────────────


def _basecase_node_params() -> dict:
    """Return node_params that reproduce the Aspen base case
    (T_rxn=230°C, P_rxn=70 bar, f_purge=0.005, P_col=1.56 bar,
    capacity_factor=1.0).  Other node params are set to the midpoint
    of their ranges — they don't feed the KPI pipeline today."""
    params = {
        nid: [(lo + hi) / 2 for lo, hi in spec["ranges"]]
        for nid, spec in NODE_FEATURE_SPEC.items()
    }
    # Reactor: [T_react, P_react, WHSV, L_bed, D_bed, cat_mass_kg]
    params[6] = [230.0, 70.0, 5.0, 7.5, 2.5, 2750.0]
    # Splitter: [purge_fraction]
    params[9] = [0.005]
    # Distillation: [n_trays, RR, P_dist, T_cond, T_reb_dist]
    params[10] = [30, 3.0, 1.56, 70.0, 110.0]
    # PEM: [P_elec_MW, eta_elec, T_stack, capacity_factor] — cap=1.0
    # means F_H2 ≈ H2_FEED_KMOL = 2704 kmol/h (Aspen base case).
    params[3] = [35.0, 0.65, 75.0, 1.0]
    return params


def test_edge_features_nonzero_at_basecase() -> None:
    """At the Aspen base case every edge has non-trivial m_dot, T, P."""
    ef = compute_edge_features(_basecase_node_params())
    assert ef.shape == (13, EDGE_FEATURE_DIM)
    # Slots 0,1,2 = m_dot, T, P.  Every edge should have positive m_dot
    # and a sensible T / P.
    assert (ef[:, 0] > 0).all(), "every edge should carry mass flow at base case"
    assert (ef[:, 1] > 0).all(), "every edge should have non-zero T"
    assert (ef[:, 2] > 0).all(), "every edge should have non-zero P"


def test_edge_features_match_aspen_basecase() -> None:
    """Within 5 %, reactor / recycle / flash / distillation edges match
    the converged Aspen numbers when node_params == base case."""
    ef = compute_edge_features(_basecase_node_params())
    # Edge order matches flowsheet_graph.STREAMS.
    # name → (m_dot_kgps, T_C, P_bar)
    expected = {
        "reactor_effluent": (24.64, 230.0, 70.0),
        "cooled_effluent":  (24.64,  40.0, 70.0),
        "vapour":           (12.19,  40.0, 70.0),
        "recycle":          (12.13,  40.0, 70.0),
        "crude_meoh":       (12.29,  38.87, 1.56),
        "co2_comp":         (11.00,  40.0, 70.0),
        "h2":               ( 1.51, 138.92, 70.0),
    }
    for i, (_, _, name) in enumerate(STREAMS):
        if name not in expected:
            continue
        m_exp, T_exp, P_exp = expected[name]
        assert abs(ef[i, 0] - m_exp) / m_exp < 0.05, f"{name} m_dot off"
        assert abs(ef[i, 1] - T_exp) < 1.0,         f"{name} T off"
        assert abs(ef[i, 2] - P_exp) / P_exp < 0.05, f"{name} P off"


def test_edge_features_respond_to_design() -> None:
    """Perturbing T_rxn and P_col changes the corresponding edge
    features — proves the GNN sees design information, not a constant."""
    base = _basecase_node_params()
    ef0 = compute_edge_features(base)

    # Crank up T_rxn by 30 °C
    hot = {**base, 6: [260.0, 70.0, 5.0, 7.5, 2.5, 2750.0]}
    ef_hot = compute_edge_features(hot)
    # Reactor-side edges should now be at 260 °C.
    reactor_edges = {"preheated_feed", "reactor_effluent", "fehe_hot"}
    for i, (_, _, name) in enumerate(STREAMS):
        if name in reactor_edges:
            assert ef_hot[i, 1] == pytest.approx(260.0, abs=0.1)
            assert ef0[i, 1] != ef_hot[i, 1]

    # Bump column pressure to 3.0 bar
    col = {**base, 10: [30, 3.0, 3.0, 70.0, 110.0]}
    ef_col = compute_edge_features(col)
    # crude_meoh is the only column-pressure edge.
    for i, (_, _, name) in enumerate(STREAMS):
        if name == "crude_meoh":
            assert ef_col[i, 2] == pytest.approx(3.0, abs=1e-6)
            assert ef0[i, 2] != ef_col[i, 2]


def test_edge_feature_flags_preserved() -> None:
    """is_recycle (slot 8) and is_energy (slot 9) flags still match the
    historical structural pattern after the stream-conditions overhaul."""
    ef = compute_edge_features(_basecase_node_params())
    # Edge indices from flowsheet_graph.STREAMS:
    #   1 = lean_mea  (recycle)
    #   9 = fehe_hot  (energy)
    #  11 = recycle   (recycle)
    recycle_idx = {1, 11}
    energy_idx = {9}
    for i in range(13):
        want_recycle = 1.0 if i in recycle_idx else 0.0
        want_energy  = 1.0 if i in energy_idx  else 0.0
        assert ef[i, 8] == want_recycle, f"is_recycle wrong on edge {i}"
        assert ef[i, 9] == want_energy,  f"is_energy  wrong on edge {i}"


def test_compositions_sum_to_one() -> None:
    """Mass fractions on every edge sum to ~1 (Aspen conservation)."""
    ef = compute_edge_features(_basecase_node_params())
    # Slots 3..7 = x_CO2, x_H2, x_MeOH, x_H2O, x_inert
    sums = ef[:, 3:8].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-3), f"Composition sums off: {sums}"


def test_edge_features_synthetic_fallback(monkeypatch) -> None:
    """If stream_data.load_basecase_streams raises, compute_edge_features
    returns a structural-flag-only array (and slots 0–7 are all zero)."""
    import stream_data
    stream_data.reset_cache()

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated excel read failure")

    monkeypatch.setattr(stream_data, "load_basecase_streams", _boom)

    ef = compute_edge_features(_basecase_node_params())
    assert ef.shape == (13, EDGE_FEATURE_DIM)
    # Stream-condition slots all zero
    assert (ef[:, :8] == 0).all(), "fallback should zero out stream conditions"
    # Structural flags still on the right edges
    assert ef[1, 8] == 1.0 and ef[11, 8] == 1.0
    assert ef[9, 9] == 1.0
    # No other flags set
    recycle_idx = {1, 11}
    energy_idx = {9}
    for i in range(13):
        if i not in recycle_idx:
            assert ef[i, 8] == 0.0
        if i not in energy_idx:
            assert ef[i, 9] == 0.0

    # Reset so subsequent tests see the real base case again.
    stream_data.reset_cache()


# ─────────────────────────────────────────────────────────────────────
# CLI entry point so the file is runnable even without pytest installed.
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
