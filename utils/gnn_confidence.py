"""
Edge-state GNN confidence utilities.

The edge-state model predicts the 8 physical stream features for each
flowsheet edge.  This module loads the saved checkpoint and estimates a
scalar uncertainty for the current plant operating point using stochastic
forward passes.  The uncertainty is the mean predictive variance in the
model's normalised target space, which keeps the 8 physical features on a
common scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("gnn_confidence")

PHYSICAL_IDX = list(range(8))
STRUCTURAL_IDX = [8, 9]
DEFAULT_MODEL_NAME = "edge_state_model.pt"


@dataclass
class GNNPenaltyConfig:
    """Configuration for the edge-state GNN uncertainty penalty.

    The environment subtracts
    ``beta * max(0, uncertainty - threshold)`` from the base reward when
    this config is enabled.
    """

    beta: float = 2.0
    threshold: float = 1e-3
    enabled: bool = True
    mc_samples: int = 4
    dropout_p: float = 0.05
    max_penalty: float = 10.0
    model_path: Optional[str] = None
    device: str = "cpu"
    cache_enabled: bool = True
    cache_decimals_load: int = 3
    cache_decimals_T: int = 1
    cache_decimals_P: int = 1
    max_cache_size: int = 2048


@dataclass
class GNNConfidenceResult:
    """Prediction and uncertainty returned by :class:`GNNConfidenceEstimator`."""

    uncertainty: float
    confidence: float
    prediction_mean: np.ndarray
    prediction_variance: np.ndarray
    physical_prediction_variance: np.ndarray


def _resolve_model_path(model_path: Optional[str] = None) -> Path:
    """Resolve the edge-state checkpoint path.

    Supports both the package output location used by the GNN scripts and
    the root-level ``outputs/`` path mentioned in notebooks or reports.
    """

    pkg_dir = Path(__file__).resolve().parents[1]
    root_dir = pkg_dir.parent

    if model_path:
        raw = Path(model_path)
        raw = raw / DEFAULT_MODEL_NAME if raw.suffix != ".pt" else raw
        candidates = [raw]
        if not raw.is_absolute():
            candidates.extend([pkg_dir / raw, root_dir / raw])
    else:
        candidates = [
            pkg_dir / "outputs" / "gnn_sweep" / DEFAULT_MODEL_NAME,
            root_dir / "outputs" / "gnn_sweep" / DEFAULT_MODEL_NAME,
        ]

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        checked.append(str(candidate))
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Edge-state GNN checkpoint not found. Checked: " + ", ".join(checked)
    )


def _make_edge_state_predictor_class():
    """Create the checkpoint-compatible model class lazily."""

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GINEConv

    class EdgeStatePredictor(nn.Module):
        """GINE trunk with a per-edge physical-state head."""

        def __init__(
            self,
            node_dim: int,
            struct_dim: int,
            hidden_dim: int,
            n_layers: int,
            out_dim: int,
            mc_dropout_p: float = 0.0,
        ) -> None:
            super().__init__()
            self.node_embed = nn.Linear(node_dim, hidden_dim)
            self.edge_embed = nn.Linear(struct_dim, hidden_dim)

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

            head_in = 2 * hidden_dim + struct_dim
            self.edge_head = nn.Sequential(
                nn.Linear(head_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, out_dim),
            )
            self.n_layers = n_layers
            self.mc_dropout_p = float(mc_dropout_p)
            self.mc_dropout_enabled = False

        def _maybe_dropout(self, x: torch.Tensor) -> torch.Tensor:
            if self.mc_dropout_p <= 0.0:
                return x
            return F.dropout(
                x,
                p=self.mc_dropout_p,
                training=self.mc_dropout_enabled,
            )

        def forward(self, data: Any) -> torch.Tensor:
            x = self.node_embed(data.x)
            e = self.edge_embed(data.edge_attr)

            for i in range(self.n_layers):
                x_res = x
                x = self.convs[i](x, data.edge_index, e)
                x = self.bns[i](x)
                x = F.relu(x + x_res)
                x = self._maybe_dropout(x)

            src, dst = data.edge_index
            edge_repr = torch.cat([x[src], x[dst], data.edge_attr], dim=-1)
            edge_repr = self._maybe_dropout(edge_repr)
            return self.edge_head(edge_repr)

    return EdgeStatePredictor


def plant_state_to_node_params(
    state: Optional[Mapping[str, float]] = None,
    *,
    load: Optional[float] = None,
    T: Optional[float] = None,
    P: Optional[float] = None,
) -> Dict[int, list[float]]:
    """Map the RL environment state onto the flowsheet GNN design vector.

    The edge-state GNN was trained on static flowsheet design parameters.
    For dynamic control, only the controllable operating knobs are updated:
    electrolyser capacity factor plus reactor temperature and pressure.
    All other node parameters are held at the midpoint of their training
    ranges.
    """

    from .. import flowsheet_graph as fg

    params: Dict[int, list[float]] = {}
    for node_id, spec in fg.NODE_FEATURE_SPEC.items():
        params[node_id] = [
            0.5 * (float(lo) + float(hi)) for lo, hi in spec["ranges"]
        ]

    def pick(explicit: Optional[float], *keys: str, default: float) -> float:
        if explicit is not None:
            return float(explicit)
        if state is not None:
            for key in keys:
                if key in state:
                    return float(state[key])
        return float(default)

    load_val = pick(load, "load", "electrolyser_load", default=params[3][3])
    T_val = pick(T, "T", "reactor_T", default=params[6][0])
    P_val = pick(P, "P", "reactor_P", default=params[6][1])

    # Node 3: PEM electrolyser [P_elec_MW, eta_elec, T_stack, capacity_factor]
    params[3][3] = load_val

    # Pressure-related upstream/downstream units track reactor pressure.
    params[2][0] = P_val  # compressor outlet pressure
    params[6][0] = T_val
    params[6][1] = P_val
    params[8][1] = P_val  # flash pressure

    return params


def build_edge_state_graph(
    state: Optional[Mapping[str, float]] = None,
    *,
    load: Optional[float] = None,
    T: Optional[float] = None,
    P: Optional[float] = None,
) -> Any:
    """Build a PyG graph for edge-state inference from plant state."""

    from .. import flowsheet_graph as fg

    node_params = plant_state_to_node_params(state, load=load, T=T, P=P)
    edge_features = np.zeros((len(fg.STREAMS), fg.EDGE_FEATURE_DIM), dtype=float)
    try:
        edge_features = fg._structural_flags(edge_features)
    except AttributeError:
        for edge_idx in (1, 11):
            edge_features[edge_idx, 8] = 1.0
        edge_features[9, 9] = 1.0

    data = fg.build_base_graph(node_params=node_params, edge_features=edge_features)
    data.edge_attr = data.edge_attr[:, STRUCTURAL_IDX]
    return data


class GNNConfidenceEstimator:
    """Lazy loader and inference wrapper for the edge-state GNN."""

    def __init__(self, config: Optional[GNNPenaltyConfig] = None) -> None:
        self.config = config or GNNPenaltyConfig()
        self._model: Any = None
        self._torch: Any = None
        self._device: Any = None
        self._mu_y: Any = None
        self._sig_y: Any = None
        self._model_path: Optional[Path] = None
        self._cache: Dict[Tuple[float, float, float], GNNConfidenceResult] = {}

    def load(self) -> "GNNConfidenceEstimator":
        """Load the checkpoint if it has not already been loaded."""

        if self._model is not None:
            return self

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Edge-state GNN confidence requires PyTorch. Install the "
                "GNN dependencies before enabling use_gnn_penalty."
            ) from exc

        self._torch = torch
        requested_device = self.config.device
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"

        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested for GNN confidence but unavailable; using CPU")
            device = torch.device("cpu")
        else:
            device = torch.device(requested_device)
        self._device = device

        model_path = _resolve_model_path(self.config.model_path)
        ckpt = torch.load(model_path, map_location=device)

        struct_dim = len(ckpt.get("struct_features", STRUCTURAL_IDX))
        out_dim = len(ckpt.get("phys_features", PHYSICAL_IDX))
        try:
            predictor_cls = _make_edge_state_predictor_class()
        except ModuleNotFoundError as exc:
            if exc.name == "torch_geometric":
                raise RuntimeError(
                    "Edge-state GNN confidence requires torch_geometric. "
                    "Install the GNN dependencies before enabling "
                    "use_gnn_penalty."
                ) from exc
            raise
        model = predictor_cls(
            node_dim=int(ckpt["node_dim"]),
            struct_dim=struct_dim,
            hidden_dim=int(ckpt["hidden_dim"]),
            n_layers=int(ckpt["n_layers"]),
            out_dim=out_dim,
            mc_dropout_p=float(self.config.dropout_p),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()

        self._model = model
        self._mu_y = ckpt["mu_y"].detach().to(device)
        self._sig_y = ckpt["sig_y"].detach().to(device)
        self._model_path = model_path
        LOGGER.info("Loaded edge-state GNN confidence model from %s", model_path)
        return self

    def estimate(
        self,
        state: Optional[Mapping[str, float]] = None,
        *,
        load: Optional[float] = None,
        T: Optional[float] = None,
        P: Optional[float] = None,
    ) -> GNNConfidenceResult:
        """Estimate edge-state prediction uncertainty for one plant state."""

        self.load()
        assert self._torch is not None
        assert self._model is not None
        assert self._device is not None
        assert self._mu_y is not None
        assert self._sig_y is not None

        key = self._cache_key(state, load=load, T=T, P=P)
        if key is not None and key in self._cache:
            return self._cache[key]

        data = build_edge_state_graph(state, load=load, T=T, P=P)
        data = data.to(self._device)

        samples = max(1, int(self.config.mc_samples))
        stochastic = samples > 1 and self.config.dropout_p > 0.0
        preds = []

        self._model.eval()
        self._model.mc_dropout_enabled = stochastic
        with self._torch.no_grad():
            for _ in range(samples):
                preds.append(self._model(data).detach())
        self._model.mc_dropout_enabled = False

        pred_stack = self._torch.stack(preds, dim=0)
        pred_mean_norm = pred_stack.mean(dim=0)
        pred_var_norm = pred_stack.var(dim=0, unbiased=False)

        pred_mean_phys = pred_mean_norm * self._sig_y + self._mu_y
        pred_var_phys = pred_var_norm * (self._sig_y ** 2)

        uncertainty = float(pred_var_norm.mean().item())
        confidence = float(1.0 / (1.0 + uncertainty))
        result = GNNConfidenceResult(
            uncertainty=uncertainty,
            confidence=confidence,
            prediction_mean=pred_mean_phys.cpu().numpy(),
            prediction_variance=pred_var_norm.cpu().numpy(),
            physical_prediction_variance=pred_var_phys.cpu().numpy(),
        )

        if key is not None:
            if len(self._cache) >= max(1, int(self.config.max_cache_size)):
                self._cache.clear()
            self._cache[key] = result
        return result

    def _cache_key(
        self,
        state: Optional[Mapping[str, float]],
        *,
        load: Optional[float],
        T: Optional[float],
        P: Optional[float],
    ) -> Optional[Tuple[float, float, float]]:
        if not self.config.cache_enabled:
            return None

        def get(explicit: Optional[float], *keys: str) -> Optional[float]:
            if explicit is not None:
                return float(explicit)
            if state is not None:
                for key in keys:
                    if key in state:
                        return float(state[key])
            return None

        load_val = get(load, "load", "electrolyser_load")
        T_val = get(T, "T", "reactor_T")
        P_val = get(P, "P", "reactor_P")
        if load_val is None or T_val is None or P_val is None:
            return None

        return (
            round(load_val, self.config.cache_decimals_load),
            round(T_val, self.config.cache_decimals_T),
            round(P_val, self.config.cache_decimals_P),
        )


_ESTIMATOR_CACHE: Dict[Tuple[Any, ...], GNNConfidenceEstimator] = {}


def get_gnn_confidence_estimator(
    config: Optional[GNNPenaltyConfig] = None,
) -> GNNConfidenceEstimator:
    """Return a cached estimator for scripts that do not manage one."""

    cfg = config or GNNPenaltyConfig()
    key = (
        cfg.model_path,
        cfg.device,
        cfg.mc_samples,
        cfg.dropout_p,
        cfg.threshold,
        cfg.beta,
    )
    if key not in _ESTIMATOR_CACHE:
        _ESTIMATOR_CACHE[key] = GNNConfidenceEstimator(cfg)
    return _ESTIMATOR_CACHE[key]


def compute_gnn_penalty(
    load: float,
    T: float,
    P: float,
    config: Optional[GNNPenaltyConfig] = None,
    estimator: Optional[GNNConfidenceEstimator] = None,
) -> Tuple[float, float, float]:
    """Compute the scalar GNN uncertainty penalty for one operating point.

    Returns
    -------
    penalty : float
        Non-negative reward penalty.
    uncertainty : float
        Mean normalised predictive variance from stochastic edge-state
        inference.
    confidence : float
        Convenience score in (0, 1], defined as ``1 / (1 + uncertainty)``.
    """

    cfg = config or GNNPenaltyConfig()
    if not cfg.enabled:
        return 0.0, 0.0, 1.0

    est = estimator or get_gnn_confidence_estimator(cfg)
    result = est.estimate(load=load, T=T, P=P)
    if not np.isfinite(result.uncertainty):
        return 0.0, 0.0, 1.0

    excess = max(0.0, result.uncertainty - float(cfg.threshold))
    penalty = float(cfg.beta) * excess
    if not np.isfinite(penalty):
        penalty = float(cfg.max_penalty)
    else:
        penalty = min(float(penalty), float(cfg.max_penalty))
    return float(penalty), float(result.uncertainty), float(result.confidence)
