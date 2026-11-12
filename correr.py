"""Orquestador principal (1 solo comando): DuckDB/Parquet -> Tarea A -> Tarea B -> Dashboard -> costo.json."""
import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.common import TELEMETRIA, fijar_semillas, hardware, ingerir, tamano_dir
from pipeline.coordinacion import detectar_coordinacion
from pipeline.dashboard import generar_dashboard
from pipeline.narrativas import procesar_narrativas


def main():
    fijar_semillas()
    ap = argparse.ArgumentParser(description="Pipeline determinista QualyData (Tarea A + Tarea B)")
    ap.add_argument("--datos", type=Path, default=Path("datos/practica"))
    ap.add_argument("--entrega", type=Path, default=Path("salida/practica"))
    ap.add_argument("--cache", type=Path, default=Path("salida/cache"))
    args = ap.parse_args()

    t_inicio = time.perf_counter()
    datos_dir = args.datos.resolve()
    entrega_dir = args.entrega.resolve()
    cache_dir = args.cache.resolve()
    db_dir = cache_dir / "db"

    entrega_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingesta columnar a Parquet (zstd) + DuckDB en disco, leido con Polars
    con = ingerir(datos_dir, db_dir)
    df_com_raw = pl.read_parquet(db_dir / "comentarios.parquet").sort("comentario_id").to_pandas()
    df_cta_raw = pl.read_parquet(db_dir / "cuentas.parquet").sort("cuenta_id").to_pandas()

    # 2. Tarea A: Narrativas, origen (changepoint) y postura
    df_com_out, df_narr_out = procesar_narrativas(df_com_raw, cache_dir)

    # 3. Tarea B: Operaciones coordinadas via grafo en DuckDB + Leiden
    df_cta_out, aristas = detectar_coordinacion(con, df_com_raw, df_cta_raw)

    # Exportar los 4 archivos oficiales con Polars
    pl.from_pandas(df_com_out).write_csv(entrega_dir / "comentarios.csv")
    pl.from_pandas(df_narr_out).write_csv(entrega_dir / "narrativas.csv")
    pl.from_pandas(df_cta_out).write_csv(entrega_dir / "cuentas.csv")

    seg = round(time.perf_counter() - t_inicio, 2)
    n_com = len(df_com_raw)
    usd = round(TELEMETRIA["usd_ia"], 6)
    costo = {
        "comentarios_procesados": n_com,
        "segundos": seg,
        "tokens_ia": TELEMETRIA["tokens_ia"],
        "usd_ia": usd,
        "usd_por_mil_comentarios": round(usd / max(n_com / 1000.0, 1e-6), 6),
        "hardware": hardware(),
        "almacenamiento": f"DuckDB + Parquet zstd ({round(tamano_dir(db_dir) / 2**20, 2)} MB en disco)",
        "ram_pico_mb": round(TELEMETRIA["ram_pico_mb"], 1),
        "vram_pico_mb": round(TELEMETRIA["vram_pico_mb"], 1),
        "etapas": TELEMETRIA["etapas"],
        "notas": "Embeddings locales FP16 (mpnet) + Leiden GPU k-NN + grafo respuesta_a + DuckDB CIB",
    }
    (entrega_dir / "costo.json").write_text(json.dumps(costo, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4. Dashboard estatico interactivo fuera de entrega/ (entrega/ lleva exactamente los 4 archivos oficiales)
    dash_path = ROOT / "docs" / "dashboard.html"
    dash_path.parent.mkdir(parents=True, exist_ok=True)
    generar_dashboard(df_com_raw, df_com_out, df_narr_out, df_cta_out, aristas, costo, dash_path)
    old_dash = entrega_dir / "dashboard.html"
    if old_dash.exists():
        old_dash.unlink()
    con.close()

    print(
        f"\n[OK] Listo en {seg}s | {len(df_narr_out)} narrativas | "
        f"{int((df_cta_out['grupo'] != '').sum())} cuentas coordinadas ({df_cta_out['grupo'].nunique() - 1} ops) | "
        f"Entrega en {entrega_dir}"
    )


if __name__ == "__main__":
    main()
