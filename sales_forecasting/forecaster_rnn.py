from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn

from .config import get_settings, safe_id_for_path


_settings = get_settings()


def _model_path(product_id: str) -> Path:
    sid = safe_id_for_path(product_id)
    return _settings.model_dir / f"rnn_product_{sid}.pt"


class SalesLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


@dataclass
class RNNConfig:
    window_size: int = 28
    epochs: int = 30
    lr: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 2


def _create_dataset(y: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    X, t = [], []
    for i in range(len(y) - window_size):
        X.append(y[i : i + window_size])
        t.append(y[i + window_size])
    if not X:
        return np.empty((0, window_size, 1), dtype=float), np.empty((0, 1), dtype=float)
    X = np.array(X, dtype=float)[..., None]
    t = np.array(t, dtype=float)[:, None]
    return X, t


def train_and_save(y: np.ndarray, product_id: str, cfg: RNNConfig | None = None) -> Path:
    if cfg is None:
        cfg = RNNConfig()
    if y.size < cfg.window_size + 1:
        raise ValueError("Serie demasiado corta para entrenar la RNN.")

    X, t = _create_dataset(y, cfg.window_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SalesLSTM(1, cfg.hidden_size, cfg.num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    t_t = torch.tensor(t, dtype=torch.float32, device=device)

    model.train()
    for _ in range(cfg.epochs):
        opt.zero_grad(set_to_none=True)
        preds = model(X_t)
        loss = loss_fn(preds, t_t)
        loss.backward()
        opt.step()

    path = _model_path(product_id)
    payload = {
        "state_dict": model.state_dict(),
        "cfg": cfg,
    }
    torch.save(payload, path)
    return path


def load_model(product_id: str) -> tuple[SalesLSTM, RNNConfig, torch.device]:
    path = _model_path(product_id)
    if not path.exists():
        raise FileNotFoundError(f"Modelo RNN no encontrado para product_id={product_id}: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=device)
    cfg: RNNConfig = payload["cfg"]
    model = SalesLSTM(1, cfg.hidden_size, cfg.num_layers).to(device)
    model.load_state_dict(payload["state_dict"])  # type: ignore[arg-type]
    model.eval()
    return model, cfg, device


def forecast(product_id: str, horizon_days: int, history: np.ndarray) -> np.ndarray:
    model, cfg, device = load_model(product_id)
    if history.size < cfg.window_size:
        raise ValueError("Historial insuficiente para la ventana de la RNN.")

    window = history[-cfg.window_size :].astype(float)
    preds = []
    for _ in range(int(horizon_days)):
        x = torch.tensor(window[None, :, None], dtype=torch.float32, device=device)
        y_hat = model(x).detach().cpu().numpy().ravel()[0]
        y_hat = max(y_hat, 0.0)
        preds.append(y_hat)
        window = np.roll(window, -1)
        window[-1] = y_hat
    return np.array(preds, dtype=float)
