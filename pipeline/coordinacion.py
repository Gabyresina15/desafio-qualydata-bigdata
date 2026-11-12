"""Tarea B: Deteccion de operaciones coordinadas via grafo bipartito en DuckDB + Leiden."""
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd

from pipeline.common import medir, normalizar


@medir("coordinacion_duckdb_leiden")
def detectar_coordinacion(con, df_comentarios: pd.DataFrame, df_cuentas: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Detecta operaciones coordinadas sobre la conexion DuckDB y devuelve (cuentas_out, aristas_red)."""
    c = df_comentarios[["comentario_id", "publicacion_id", "cuenta_id", "fecha", "texto", "respuesta_a"]].copy()
    c["norm"] = c["texto"].map(normalizar)
    c["ts"] = pd.to_datetime(c["fecha"], utc=True).dt.as_unit("s").astype("int64")
    c["nwords"] = c["norm"].str.split().str.len()

    t_min_data = pd.to_datetime(c["fecha"], utc=True).min()
    cu = df_cuentas[["cuenta_id", "fecha_creacion"]].copy()
    cu["fc_dt"] = pd.to_datetime(cu["fecha_creacion"], utc=True)
    cu["age_d"] = (t_min_data - cu["fc_dt"]).dt.total_seconds() / 86400.0

    con.register("tmp_com_coord", c[["comentario_id", "publicacion_id", "cuenta_id", "ts", "norm", "nwords", "respuesta_a"]])
    con.register("tmp_cta_coord", cu[["cuenta_id", "age_d"]])

    pairs = con.execute("""
        WITH b AS (
            SELECT comentario_id, publicacion_id, cuenta_id, ts, norm, nwords,
                   (ts // 1800) AS b1,
                   ((ts + 900) // 1800) AS b2,
                   COUNT(*) OVER (PARTITION BY publicacion_id, ts // 1800) AS win_vol
            FROM tmp_com_coord
        ),
        co1 AS (
            SELECT a.cuenta_id AS u, b.cuenta_id AS v, a.publicacion_id AS pid,
                   1.0 / (1.0 + ln(1.0 + a.win_vol) * ln(1.0 + a.win_vol)) AS w,
                   CASE WHEN a.nwords >= 5 AND b.nwords >= 5 AND a.norm = b.norm THEN 1 ELSE 0 END AS exact_txt
            FROM b a
            JOIN b ON a.publicacion_id = b.publicacion_id
                AND a.b1 = b.b1
                AND a.cuenta_id < b.cuenta_id
                AND ABS(a.ts - b.ts) <= 1200
            UNION ALL
            SELECT a.cuenta_id AS u, b.cuenta_id AS v, a.publicacion_id AS pid,
                   1.0 / (1.0 + ln(1.0 + a.win_vol) * ln(1.0 + a.win_vol)) AS w,
                   CASE WHEN a.nwords >= 5 AND b.nwords >= 5 AND a.norm = b.norm THEN 1 ELSE 0 END AS exact_txt
            FROM b a
            JOIN b ON a.publicacion_id = b.publicacion_id
                AND a.b2 = b.b2
                AND a.cuenta_id < b.cuenta_id
                AND ABS(a.ts - b.ts) <= 1200
        ),
        co AS (
            SELECT u, v, pid, MAX(w) AS w, MAX(exact_txt) AS exact_txt
            FROM co1
            GROUP BY 1, 2, 3
        ),
        agg_co AS (
            SELECT u, v, COUNT(*) AS k_sync, SUM(w) AS sum_w, SUM(exact_txt) AS n_exact
            FROM co
            GROUP BY 1, 2
            HAVING COUNT(*) >= 2
        ),
        rep AS (
            SELECT LEAST(c1.cuenta_id, c2.cuenta_id) AS u,
                   GREATEST(c1.cuenta_id, c2.cuenta_id) AS v,
                   COUNT(*) AS n_rep
            FROM tmp_com_coord c1
            JOIN tmp_com_coord c2 ON c1.respuesta_a = c2.comentario_id
            WHERE c1.respuesta_a <> '' AND c1.cuenta_id <> c2.cuenta_id
            GROUP BY 1, 2
        )
        SELECT a.u, a.v, a.k_sync, a.sum_w, a.n_exact, COALESCE(r.n_rep, 0) AS n_rep,
               cu1.age_d AS age_u, cu2.age_d AS age_v
        FROM agg_co a
        LEFT JOIN rep r ON a.u = r.u AND a.v = r.v
        JOIN tmp_cta_coord cu1 ON a.u = cu1.cuenta_id
        JOIN tmp_cta_coord cu2 ON a.v = cu2.cuenta_id
        ORDER BY a.u, a.v
    """).df()

    acc_list = sorted(cu["cuenta_id"].unique())
    a2i = {a: i for i, a in enumerate(acc_list)}
    n_acc = len(acc_list)

    u_idx = pairs["u"].map(a2i).to_numpy()
    v_idx = pairs["v"].map(a2i).to_numpy()
    k_arr = pairs["k_sync"].to_numpy()
    w_arr = pairs["sum_w"].to_numpy()
    ex_arr = pairs["n_exact"].to_numpy()
    rep_arr = pairs["n_rep"].to_numpy()
    both_new = (pairs["age_u"].to_numpy() < 30) & (pairs["age_v"].to_numpy() < 30)

    mask = (k_arr >= 3) | (both_new & (k_arr >= 2))
    edges = list(zip(u_idx[mask].tolist(), v_idx[mask].tolist()))
    wts = (w_arr[mask] + 0.5 * ex_arr[mask] + 0.3 * rep_arr[mask] + 0.4 * both_new[mask]).tolist()

    g = ig.Graph(n=n_acc, edges=edges, directed=False)
    g.es["weight"] = wts
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=1.0, seed=42
    )
    mem = np.array(part.membership)
    vals, cnts = np.unique(mem, return_counts=True)

    pred = {}
    gid = 1
    flagged_nodes = set()
    for v_id, cnt in sorted(zip(vals, cnts), key=lambda x: (-x[1], x[0])):
        if cnt < 8 or cnt > 400:
            continue
        nodes = np.where(mem == v_id)[0]
        sub = g.subgraph(nodes.tolist())
        avg_deg = 2.0 * sub.ecount() / len(nodes)
        dens = sub.density()
        mean_w = float(np.mean(sub.es["weight"])) if sub.ecount() else 0.0
        if avg_deg >= 5.0 and dens >= 0.35 and mean_w >= 0.30:
            lbl = f"g{gid:03d}"
            gid += 1
            for ni in nodes:
                pred[acc_list[ni]] = lbl
                flagged_nodes.add(acc_list[ni])

    out = pd.DataFrame({"cuenta_id": acc_list, "grupo": [pred.get(a, "") for a in acc_list]})

    # Aristas internas de las operaciones detectadas para el dashboard
    aristas = []
    for u_s, v_s, k_s, w_s in zip(
        pairs.loc[mask, "u"], pairs.loc[mask, "v"], pairs.loc[mask, "k_sync"], wts
    ):
        if u_s in flagged_nodes and v_s in flagged_nodes and pred.get(u_s) == pred.get(v_s):
            aristas.append({"source": u_s, "target": v_s, "grupo": pred[u_s], "k": int(k_s), "w": round(float(w_s), 2)})
    return out, aristas[:1500]
