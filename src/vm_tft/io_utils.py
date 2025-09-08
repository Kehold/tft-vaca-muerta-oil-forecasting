

from __future__ import annotations
import numpy as np
import torch
import os, random
import json, time, shutil
from typing import Any, Optional
from pathlib import Path
from .cfg import ARTIFACTS

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def save_json(obj: Any, path: Path):
    ensure_dir(path.parent); path.write_text(json.dumps(obj, indent=2))

def set_seeds(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    try:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def device_hint() -> str:
    return "gpu" if torch.cuda.is_available() else "cpu"

def artifact_path(*parts) -> Path:
    return ensure_dir(ARTIFACTS / Path(*parts))

def runs_root(kind: str) -> Path:
    p = Path(ARTIFACTS) / "runs" / kind
    p.mkdir(parents=True, exist_ok=True)
    return p

def new_run_dir(kind: str, tag: Optional[str] = None) -> Path:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{ts}__{tag}" if tag else ts
    p = runs_root(kind) / name
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    
def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None
    
def copy_as_pointer(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    
def update_best_if_better(
    candidate_metric: float,
    best_json_path: Path,
    candidate_payload: dict,
    lower_is_better: bool = True,
) -> bool:
    current = read_json(best_json_path) or {}
    current_metric = current.get("metric_value")
    improved = (
        current_metric is None
        or (candidate_metric < current_metric if lower_is_better else candidate_metric > current_metric)
    )
    if improved:
        write_json(best_json_path, candidate_payload | {"metric_value": candidate_metric})
        return True
    return False
