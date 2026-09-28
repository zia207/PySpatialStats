"""
spatialstats.gwmodel.deep.gtnnwr
=====================
Geographically and Temporally Neural Network Weighted Regression (GTNNWR).

Extends GNNWR to the spatiotemporal domain via a Spatiotemporal Proximity
Neural Network (STPNN).

Based on: Wu et al. (2021) IJGIS 35(3), 582-608.
"""

import numpy as np
from typing import Optional, List

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from spatialstats.gwmodel.core.base import GWRegressor, _parse_geometry
from spatialstats.gwmodel.utils.distance import pairwise_distances, normalize_coords
from spatialstats.gwmodel.utils.compute import resolve_torch_device, configure_torch_threads, get_compute_config


class _STPNN(nn.Module if TORCH_AVAILABLE else object):
    """
    Spatiotemporal Proximity Neural Network.

    Fuses spatial and temporal proximity representations to produce
    space-time weight scalars.
    """

    def __init__(self, spatial_layers, temporal_layers, fusion_layers, dropout=0.1):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required: pip install torch")
        super().__init__()

        def make_mlp(dims, in_dim):
            layers = []
            d = in_dim
            for h in dims:
                layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
                d = h
            return nn.Sequential(*layers), d

        self.spatial_enc, s_out = make_mlp(spatial_layers, 2)
        self.temporal_enc, t_out = make_mlp(temporal_layers, 1)

        fusion_in = s_out + t_out
        self.fusion_enc, f_out = make_mlp(fusion_layers, fusion_in)
        self.head = nn.Sequential(nn.Linear(f_out, 1), nn.Softplus())

    def forward(self, spatial_input, temporal_input):
        s_feat = self.spatial_enc(spatial_input)
        t_feat = self.temporal_enc(temporal_input)
        fused = torch.cat([s_feat, t_feat], dim=-1)
        return self.head(self.fusion_enc(fused))


class GTNNWR(GWRegressor):
    """
    Geographically and Temporally Neural Network Weighted Regression (GTNNWR).

    Parameters
    ----------
    spatial_layers : list of int
    temporal_layers : list of int
    fusion_layers : list of int
    dropout : float
    learning_rate : float
    n_epochs : int
    batch_size : int
    patience : int
    device : str

    Examples
    --------
    >>> from spatialstats.gwmodel.deep import GTNNWR
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> n = 80
    >>> coords = np.random.rand(n, 2) * 100
    >>> times = np.random.rand(n) * 10
    >>> X = np.random.rand(n, 2)
    >>> y = X @ [1, -1] + 0.1 * times + np.random.randn(n) * 0.2
    >>> model = GTNNWR(n_epochs=10)
    >>> model.fit(X, y, coords, times)
    >>> print(model.predict(X, coords, times).shape)
    (80,)
    """

    def __init__(self, spatial_layers=[64, 32], temporal_layers=[32, 16],
                 fusion_layers=[64, 32], dropout=0.1,
                 learning_rate=1e-3, weight_decay=1e-4,
                 n_epochs=100, batch_size=64, val_ratio=0.1,
                 patience=20, device="auto", random_state=None):
        super().__init__(bandwidth=None, n_jobs=1)
        self.spatial_layers = spatial_layers
        self.temporal_layers = temporal_layers
        self.fusion_layers = fusion_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.stpnn_: Optional[object] = None
        self.train_loss_: List[float] = []

    def _get_device(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required: pip install torch")
        cfg = get_compute_config()
        choice = self.device if self.device != "auto" else cfg.device
        configure_torch_threads(cfg.resolved_torch_threads())
        return resolve_torch_device(choice)

    def fit(self, X, y, geometry, times):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required: pip install torch")
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        times = np.asarray(times, dtype=float).ravel()
        coords = _parse_geometry(geometry)
        n, p = X.shape
        X_int = np.column_stack([np.ones(n), X])
        p_full = X_int.shape[1]

        self.coords_ = coords
        self.times_ = times
        self.X_ = X_int
        self.y_ = y

        device = self._get_device()
        coords_norm = normalize_coords(coords)
        times_norm = (times - times.min()) / max(times.max() - times.min(), 1e-6)

        stpnn = _STPNN(self.spatial_layers, self.temporal_layers,
                       self.fusion_layers, self.dropout).to(device)
        optim_fn = optim.Adam(stpnn.parameters(), lr=self.learning_rate,
                              weight_decay=self.weight_decay)
        mse = nn.MSELoss()

        cten = torch.tensor(coords_norm, dtype=torch.float32).to(device)
        tten = torch.tensor(times_norm.reshape(-1, 1), dtype=torch.float32).to(device)
        Xten = torch.tensor(X_int, dtype=torch.float32).to(device)
        yten = torch.tensor(y, dtype=torch.float32).to(device)

        n_val = max(1, int(n * self.val_ratio))
        idx = np.random.permutation(n)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        best_val, patience_cnt, best_state = np.inf, 0, None

        for epoch in range(self.n_epochs):
            stpnn.train()
            # Compute ST weights for training points
            w_all = stpnn(cten[train_idx], tten[train_idx]).squeeze(-1)
            w_all = w_all / (w_all.sum() + 1e-12) * len(train_idx)
            W = torch.diag(w_all)
            Xt = Xten[train_idx]
            yt = yten[train_idx]
            XtW = Xt.T @ W
            A = XtW @ Xt + 1e-6 * torch.eye(p_full, device=device)
            b = XtW @ yt
            try:
                coef = torch.linalg.solve(A, b)
            except Exception:
                coef = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)
            loss = mse(Xt @ coef, yt)
            optim_fn.zero_grad()
            loss.backward()
            optim_fn.step()
            self.train_loss_.append(float(loss.item()))

            stpnn.eval()
            with torch.no_grad():
                val_pred = Xten[val_idx] @ coef
                val_loss = float(mse(val_pred, yten[val_idx]).item())
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in stpnn.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    break

        if best_state:
            stpnn.load_state_dict(best_state)
        self.stpnn_ = stpnn

        # Per-location coefficients
        stpnn.eval()
        self.coef_ = np.zeros((n, p_full))
        with torch.no_grad():
            for i in range(n):
                rel_s = cten - cten[i]
                rel_t = tten - tten[i]
                w_i = stpnn(rel_s, rel_t).squeeze(-1).cpu().numpy()
                w_i = w_i / max(w_i.sum(), 1e-12) * n
                W_i = np.diag(w_i)
                XtW_i = X_int.T @ W_i
                A_i = XtW_i @ X_int + 1e-6 * np.eye(p_full)
                b_i = XtW_i @ y
                try:
                    self.coef_[i] = np.linalg.solve(A_i, b_i)
                except np.linalg.LinAlgError:
                    self.coef_[i] = np.linalg.lstsq(A_i, b_i, rcond=None)[0]

        self.fitted_ = np.array([X_int[i] @ self.coef_[i] for i in range(n)])
        self.residuals_ = y - self.fitted_
        self.bandwidth_ = None
        return self

    def predict(self, X, geometry, times=None) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Fit the model first.")
        X = np.asarray(X, dtype=float)
        coords_new = _parse_geometry(geometry)
        m = len(coords_new)
        X_int = np.column_stack([np.ones(m), X])
        from scipy.spatial.distance import cdist
        nearest = np.argmin(cdist(coords_new, self.coords_), axis=1)
        return np.array([X_int[j] @ self.coef_[nearest[j]] for j in range(m)])

    def get_params(self, deep=True):
        return dict(spatial_layers=self.spatial_layers,
                    temporal_layers=self.temporal_layers,
                    fusion_layers=self.fusion_layers,
                    dropout=self.dropout,
                    learning_rate=self.learning_rate,
                    n_epochs=self.n_epochs, batch_size=self.batch_size,
                    patience=self.patience, device=self.device)
