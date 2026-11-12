"""Semillas, telemetría (decorador @medir) y capa de almacenamiento DuckDB/Parquet."""
import os

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # requerido por use_deterministic_algorithms en CUDA

import functools
import json
import random
import re
import time
import unicodedata
from pathlib import Path

import numpy as np

SEED = 42


def fijar_semillas():
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------- telemetría
TELEMETRIA = {"etapas": {}, "tokens_ia": 0, "usd_ia": 0.0, "ram_pico_mb": 0.0, "vram_pico_mb": 0.0}


def _ram_mb():
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def medir(nombre):
    """Decorador: tiempo exacto, pico de RAM/VRAM por etapa."""
    def deco(fn):
        @functools.wraps(fn)
        def envoltura(*a, **k):
            vram0 = 0
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except ImportError:
                torch = None
            t0 = time.perf_counter()
            r = fn(*a, **k)
            dt = time.perf_counter() - t0
            vram = 0.0
            if torch is not None and torch.cuda.is_available():
                vram = torch.cuda.max_memory_allocated() / 2**20
            ram = _ram_mb()
            TELEMETRIA["etapas"][nombre] = {"segundos": round(dt, 2), "ram_mb": round(ram), "vram_pico_mb": round(vram)}
            TELEMETRIA["ram_pico_mb"] = max(TELEMETRIA["ram_pico_mb"], ram)
            TELEMETRIA["vram_pico_mb"] = max(TELEMETRIA["vram_pico_mb"], vram)
            print(f"[{nombre}] {dt:.1f}s  RAM={ram:.0f}MB  VRAM pico={vram:.0f}MB", flush=True)
            return r
        return envoltura
    return deco


def registrar_tokens(tokens, usd):
    TELEMETRIA["tokens_ia"] += int(tokens)
    TELEMETRIA["usd_ia"] += float(usd)


def tamano_dir(path):
    p = Path(path)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def hardware():
    import platform
    import psutil
    gpu = ""
    try:
        import torch
        if torch.cuda.is_available():
            gpu = f", {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory // 2**30} GB VRAM"
    except ImportError:
        pass
    return f"{platform.processor() or platform.machine()}, {psutil.cpu_count(logical=False)} cores, {psutil.virtual_memory().total // 2**30} GB RAM{gpu}"


# ---------------------------------------------------------------- texto
ABREV = {"q": "que", "xq": "porque", "x": "por", "tmb": "tambien", "pa": "para", "tdo": "todo", "d": "de"}


def normalizar(t):
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[^a-zñ0-9#@\s]", " ", t)
    t = re.sub(r"(\w)\1{2,}", r"\1\1", t)
    return " ".join(ABREV.get(w, w) for w in t.split())


# ---------------------------------------------------------------- almacenamiento
@medir("ingesta_duckdb")
def ingerir(datos: Path, db_dir: Path):
    """CSV.gz -> Parquet columnar (zstd) + base DuckDB en disco. Devuelve conexión."""
    import duckdb
    db_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_dir / "qualydata.duckdb"))
    con.execute("SET threads=8; SET preserve_insertion_order=false;")
    for nombre in ("comentarios", "cuentas", "publicaciones"):
        src = next(datos.glob(f"{nombre}.csv*"))
        pq = db_dir / f"{nombre}.parquet"
        con.execute(f"""COPY (SELECT * FROM read_csv('{src.as_posix()}', header=true, all_varchar=true,
                       ignore_errors=true, quote='"', escape='"', delim=',', null_padding=true, strict_mode=false)
                       ORDER BY 1) TO '{pq.as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
        con.execute(f"CREATE OR REPLACE TABLE {nombre} AS SELECT * FROM read_parquet('{pq.as_posix()}')")
    return con
