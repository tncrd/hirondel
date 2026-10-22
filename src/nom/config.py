"""Model configuration, read from the `config.json` shipped next to the weights."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields

FORMAT_VERSION = 1


@dataclass(frozen=True)
class FusionConfig:
    state_dim: int = 32
    spectral_bands: int = 16
    basis_size: int = 8
    hidden: int = 128
    max_horizon: int = 720


@dataclass(frozen=True)
class NomConfig:
    d_model: int = 360
    n_heads: int = 4
    n_layers: int = 6
    layer_types: tuple[str, ...] = ("delta", "attention") * 3
    ffn_hidden: int = 480
    embed_hidden: int = 240
    patch_size: int = 16
    conv_kernel: int = 4
    delta_write_max: float = 2.0
    delta_chunk: int = 16
    norm_eps: float = 1e-6
    rope_base: float = 10000.0
    time_scale: float = 8.0
    loop_passes: int = 2
    anchor_rank: int = 64
    quantile_levels: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    context_length: int = 2048
    native_horizon: int = 720
    coarse_factors: tuple[int, ...] = (4,)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    input_transform: str = "asinh"
    output_clip: float = 15.0
    min_scale: float = 1e-5
    range_margin: float = 1e4
    long_view_base_season: int = 24
    flip: bool = True
    format_version: int = FORMAT_VERSION

    @classmethod
    def from_dict(cls, d: dict) -> "NomConfig":
        if d.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported config format_version {d.get('format_version')!r} "
                             f"(this package reads version {FORMAT_VERSION})")
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        kw = dict(d)
        for name in ("layer_types", "quantile_levels", "coarse_factors"):
            if name in kw:
                kw[name] = tuple(kw[name])
        if "fusion" in kw:
            kw["fusion"] = FusionConfig(**kw["fusion"])
        cfg = cls(**kw)
        if cfg.input_transform != "asinh":
            raise ValueError(f"unsupported input_transform {cfg.input_transform!r}")
        if len(cfg.layer_types) != cfg.n_layers or cfg.n_layers < 3:
            raise ValueError("layer_types must list n_layers >= 3 entries")
        return cfg

    @classmethod
    def from_json(cls, path: str) -> "NomConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @property
    def median_index(self) -> int:
        return min(range(len(self.quantile_levels)), key=lambda i: abs(self.quantile_levels[i] - 0.5))
