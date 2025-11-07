"""Positional encoding modules for process mining models."""

import math

import torch
from torch import nn

# === Positional Encoding Classes ===

class SinusoidalPositionalEncoding(nn.Module):
    """Standard (non-learnable) sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_period: float = 10000.0) -> None:
        """Initialize the SinusoidalPositionalEncoding module."""
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute the sinusoidal positional encodings."""
        if seq_len <= 0:
            return torch.zeros(1, 0, self.d_model, device=device, dtype=dtype)
        positions = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        n_even = (self.d_model + 1) // 2
        n_odd = self.d_model // 2
        base = -math.log(self.max_period) / max(self.d_model, 1)
        div_even = torch.exp(torch.arange(n_even, device=device, dtype=dtype) * base)
        div_odd = torch.exp(torch.arange(n_odd, device=device, dtype=dtype) * base)
        pe = torch.zeros(seq_len, self.d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(positions * div_even)
        if n_odd > 0:
            pe[:, 1::2] = torch.cos(positions * div_odd)
        return pe.unsqueeze(0)


class PeriodicPositionalEncoding(nn.Module):
    """Learnable periodic positional encoding using Fourier features."""

    def __init__(self, d_model: int, n_freqs: int | None = None, min_exp: float = -6.0, max_exp: float = 0.0) -> None:
        """Initialize the PeriodicPositionalEncoding module."""
        super().__init__()
        if n_freqs is None:
            n_freqs = d_model // 2
        self.n_freqs = n_freqs
        self.d_model = d_model
        init_exps = torch.linspace(min_exp, max_exp, steps=n_freqs)
        freqs_init = (2.0 * math.pi) * torch.pow(10.0, init_exps)
        self.log_freqs = nn.Parameter(torch.log(freqs_init))
        self.phases = nn.Parameter(torch.zeros(n_freqs))
        self.proj = nn.Linear(2 * n_freqs, d_model, bias=True) if 2 * n_freqs != d_model else None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute the periodic positional encodings."""
        positions = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        freqs = torch.exp(self.log_freqs).to(device=device, dtype=dtype)
        phases = self.phases.to(device=device, dtype=dtype)
        theta = positions * freqs.unsqueeze(0) + phases.unsqueeze(0)
        feat = torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)
        if self.proj is not None:
            feat = self.proj(feat)
        return feat.unsqueeze(0)


class SharpPeriodicRelativeEncoding(nn.Module):
    """Periodic *relative* positional encoding with sharp (non-smooth) transitions."""

    def __init__(self, d_model: int, mode: str = "square", max_period: float = 10000.0) -> None:
        """Initialize the SharpPeriodicRelativeEncoding module."""
        super().__init__()
        if mode not in {"square", "sawtooth", "quantized"}:
            msg = f"Unsupported mode: {mode}"
            raise ValueError(msg)
        self.d_model = d_model
        self.mode = mode
        inv_freq = torch.exp(-math.log(max_period) * torch.arange(0, d_model, 2).float() / d_model)
        self.register_buffer("inv_freq", inv_freq)

    def _sharp_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "square":
            return torch.sign(torch.sin(x))
        if self.mode == "sawtooth":
            frac = (x / (2 * math.pi)) % 1.0
            return 2.0 * frac - 1.0
        if self.mode == "quantized":
            s = torch.sin(x)
            return torch.round(s * 2.0) / 2.0
        msg = f"Unsupported mode: {self.mode}"
        raise ValueError(msg)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute the sharp periodic relative positional encodings."""
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        rel_pos = positions[None, :] - positions[:, None]
        x = rel_pos.unsqueeze(-1) * self.inv_freq # type: ignore  # noqa: PGH003
        sharp = self._sharp_fn(x)
        return torch.cat([sharp, -sharp], dim=-1)



class LearnableRelativePositionalEncoding(nn.Module):
    """
    Differentiable learnable *relative* positional encoding with periodic features.

    Each relative offset (i - j) is mapped through learnable frequencies and phases.
    """

    def __init__(self, d_model: int, n_freqs: int | None = None, min_exp: float = -6.0, max_exp: float = 0.0) -> None:
        """Initialize the LearnableRelativePositionalEncoding module."""
        super().__init__()
        if n_freqs is None:
            n_freqs = d_model // 2
        self.n_freqs = n_freqs
        self.d_model = d_model

        init_exps = torch.linspace(min_exp, max_exp, steps=n_freqs)
        freqs_init = (2.0 * math.pi) * torch.pow(10.0, init_exps)

        self.log_freqs = nn.Parameter(torch.log(freqs_init))
        self.phases = nn.Parameter(torch.zeros(n_freqs))

        if 2 * n_freqs != d_model:
            self.proj = nn.Linear(2 * n_freqs, d_model, bias=True)
        else:
            self.proj = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute the learnable relative positional encodings."""
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        rel = positions[None, :] - positions[:, None]  # [L, L]
        freqs = torch.exp(self.log_freqs).to(device=device, dtype=dtype)
        phases = self.phases.to(device=device, dtype=dtype)

        theta = rel.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0) + phases  # [L, L, n_freqs]
        sinp = torch.sin(theta)
        cosp = torch.cos(theta)
        feat = torch.cat([sinp, cosp], dim=-1)
        if self.proj is not None:
            feat = self.proj(feat)
        return feat  # [L, L, d_model]
