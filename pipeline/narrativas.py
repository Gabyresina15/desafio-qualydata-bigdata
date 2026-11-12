"""Tarea A: Descubrimiento de narrativas, fecha de origen (changepoint), titulo/frases clave y postura."""
import hashlib
import json
import os
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from pipeline.common import medir, normalizar, registrar_tokens

STOPWORDS = sorted({
    "a", "al", "algo", "con", "como", "de", "del", "el", "ella", "en", "es", "esa", "ese", "eso",
    "esta", "este", "esto", "la", "las", "le", "lo", "los", "mas", "me", "mi", "muy", "no", "nos",
    "o", "para", "pero", "por", "que", "se", "si", "sin", "su", "sus", "te", "todo", "todos", "un",
    "una", "uno", "y", "ya", "ahi", "aca", "hay", "ver", "digo", "ser", "esta", "son", "tan",
})

REFUTE_PURE = [
    "mentira", "falso", "falsa", "fake", "verso", "nada que ver", "no es cierto", "trucho",
    "trucha", "desmintio", "sin pruebas", "inventan", "dejen de", "no es verdad", "puro cuento",
    "chamuyo", "humo", "opereta", "de otro lado", "no tiene sentido", "es viejo", "audio viejo",
]
AGREE_PURE = [
    "coincido", "tal cual", "exacto", "100%", "posta", "asi es", "clarisimo", "totalmente",
    "eso mismo", "ni mas ni menos", "lo mismo digo", "de acuerdo", "toda la razon",
]


EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBED_REV = "4328cf26390c98c5e3c738b4460a05b95f4911f5"
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
NLI_REV = "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"


@medir("embeddings_gpu")
def calcular_embeddings(textos_unicos: list[str], cache_path: Path | None = None, force_cpu: bool = False) -> np.ndarray:
    h = hashlib.sha256("\n".join(textos_unicos).encode("utf-8")).hexdigest()[:12]
    if cache_path is not None:
        meta = cache_path.with_suffix(".sha256")
        if cache_path.exists() and meta.exists() and meta.read_text().strip() == h:
            return np.load(cache_path)

    from sentence_transformers import SentenceTransformer
    device = "cpu" if (force_cpu or not torch.cuda.is_available()) else "cuda"
    try:
        model = SentenceTransformer(EMBED_MODEL, revision=EMBED_REV, device=device)
    except Exception:
        model = SentenceTransformer(EMBED_MODEL, revision=EMBED_REV, device=device, local_files_only=True)
    if device == "cuda":
        model.half()  # FP16 en GPU; FP32 automatico en CPU
    emb = model.encode(
        textos_unicos,
        batch_size=256 if device == "cuda" else 64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
        cache_path.with_suffix(".sha256").write_text(h)
    return emb


def _knn_gpu(E: np.ndarray, k: int = 30, force_cpu: bool = False) -> tuple[np.ndarray, np.ndarray]:
    n = len(E)
    device = "cpu" if (force_cpu or not torch.cuda.is_available()) else "cuda"
    dtype = torch.float16 if device == "cuda" else torch.float32
    Et = torch.from_numpy(E).to(device, dtype=dtype)
    top_vals = np.empty((n, k), dtype=np.float32)
    top_idx = np.empty((n, k), dtype=np.int32)
    bs = 2048 if device == "cuda" else 1024
    with torch.no_grad():
        for i in range(0, n, bs):
            sims = Et[i:i + bs] @ Et.T
            vals, inds = torch.topk(sims, min(k + 1, n), dim=1)
            top_vals[i:i + bs] = vals[:, 1:].float().cpu().numpy()
            top_idx[i:i + bs] = inds[:, 1:].cpu().numpy()
    return top_vals, top_idx


def _estimar_origen(dts: pd.Series, t_min_global: pd.Timestamp) -> pd.Timestamp:
    """Changepoint robusto: detecta nacimiento a mitad de periodo ignorando goteo aislado previo."""
    s = dts.sort_values()
    if len(s) < 20:
        return s.quantile(0.05)
    # Rejilla horaria desde t_min_global para comparar tasa previa vs posterior
    end_t = s.max()
    idx_hr = pd.date_range(t_min_global.floor("h"), end_t.ceil("h"), freq="1h", tz="UTC")
    hr_raw = s.to_frame("dt").assign(cnt=1).set_index("dt").resample("1h")["cnt"].sum()
    hr = hr_raw.reindex(idx_hr, fill_value=0).to_numpy(dtype=float)
    n_h = len(hr)
    if n_h < 24:
        return s.quantile(0.01)

    total = hr.sum()
    cum = np.cumsum(hr)
    min_burst = max(12.0, 0.004 * total)
    for k in range(12, n_h - 6):
        pre_mean = cum[k - 1] / k
        post_sum = total - cum[k - 1]
        post_mean = post_sum / (n_h - k)
        fwd3 = hr[k:min(k + 3, n_h)].sum()
        if pre_mean <= 0.20 * post_mean and cum[k - 1] <= 0.12 * total and fwd3 >= min_burst:
            t_cut = idx_hr[k]
            sub = s[s >= t_cut]
            if len(sub):
                return sub.iloc[0]
    return s.quantile(0.01)


def _llamar_llm_cluster(ejemplos: list[str], cache_file: Path, key: str) -> dict | None:
    """Llama 1 vez por cluster con temperature=0 y seed=42 (con cache determinista en disco)."""
    cache = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
        if key in cache:
            return cache[key]

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    prompt = (
        "Sos un analista de narrativas en redes sociales argentinas. "
        "Dado este conjunto de comentarios representativos de un mismo debate, devolve SOLO un JSON valido con:\n"
        '{"titulo": "frase corta afirmativa que resuma la tesis principal del debate", '
        '"frases_clave": ["t1", "t2", "t3", "t4", "t5"], '
        '"apoya": ["3 frases prototipo que afirman o defienden la tesis"], '
        '"rechaza": ["3 frases prototipo que niegan o refutan la tesis"]}\n\n'
        "Comentarios:\n- " + "\n- ".join(ejemplos[:15])
    )
    try:
        import urllib.request
        if os.environ.get("GEMINI_API_KEY"):
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.5-flash:generateContent?key={os.environ['GEMINI_API_KEY']}"
            )
            body = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, "seed": 42, "responseMimeType": "application/json"},
            }
            req = urllib.request.Request(
                url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                meta = data.get("usageMetadata", {})
                toks = int(meta.get("totalTokenCount", 0))
                registrar_tokens(toks, toks * 0.0000003)
                txt = data["candidates"][0]["content"]["parts"][0]["text"]
                out = json.loads(txt)
                cache[key] = out
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
                return out
        if os.environ.get("OPENAI_API_KEY"):
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps({
                    "model": "gpt-4o-mini",
                    "temperature": 0,
                    "seed": 42,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": prompt}],
                }).encode("utf-8"),
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                toks = data.get("usage", {}).get("total_tokens", 0)
                registrar_tokens(toks, toks * 0.0000003)
                out = json.loads(data["choices"][0]["message"]["content"])
                cache[key] = out
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
                return out
    except Exception:
        return None
    return None


@medir("postura_nli")
def _clasificar_postura_nli(
    c: pd.DataFrame, narr_labels: list[str], force_cpu: bool = False
) -> dict[int, str] | None:
    """Clasifica cada micro-cluster como apoya/rechaza usando NLI local contra la tesis de su narrativa."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = "cpu" if (force_cpu or not torch.cuda.is_available()) else "cuda"
        try:
            tok = AutoTokenizer.from_pretrained(NLI_MODEL, revision=NLI_REV)
            nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, revision=NLI_REV).to(device).eval()
        except Exception:
            tok = AutoTokenizer.from_pretrained(NLI_MODEL, revision=NLI_REV, local_files_only=True)
            nli = AutoModelForSequenceClassification.from_pretrained(
                NLI_MODEL, revision=NLI_REV, local_files_only=True
            ).to(device).eval()
    except Exception:
        return None

    pairs_prem, pairs_hyp, pair_mids = [], [], []
    for lbl in narr_labels:
        sub = c[c["narrativa"] == lbl]
        m_sub = sub[sub["micro"] >= 0]
        if len(m_sub) == 0:
            continue
        m_stats = m_sub.groupby("micro").agg(n=("texto", "count"), ref=("ref", "mean"))
        cands_m = m_stats[m_stats["ref"] < 0.05].sort_values("n", ascending=False)
        best_m = cands_m.index[0] if len(cands_m) else m_stats.sort_values("n", ascending=False).index[0]
        sub_bm = sub[(sub["micro"] == best_m) & (sub["nwords"] >= 6) & (~sub["ref"])]
        tesis = (
            sub_bm["texto"].value_counts().index[0]
            if len(sub_bm)
            else sub["texto"].value_counts().index[0]
        )
        for mid, gdf in m_sub.groupby("micro"):
            for ex in gdf["texto"].value_counts().head(3).index.tolist():
                pairs_prem.append(ex)
                pairs_hyp.append(tesis)
                pair_mids.append(int(mid))

    if not pairs_prem:
        return {}

    scores_m: dict[int, list[float]] = {}
    bs = 64 if device == "cuda" else 32
    with torch.no_grad():
        for i in range(0, len(pairs_prem), bs):
            inp = tok(
                pairs_prem[i:i + bs],
                pairs_hyp[i:i + bs],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            probs = torch.softmax(nli(**inp).logits, dim=-1).cpu().numpy()
            for j, mid in enumerate(pair_mids[i:i + bs]):
                scores_m.setdefault(mid, []).append(float(probs[j, 0] - probs[j, 2]))

    return {mid: ("apoya" if float(np.mean(vals)) >= 0.0 else "rechaza") for mid, vals in scores_m.items()}


@medir("narrativas_clustering")
def procesar_narrativas(df: pd.DataFrame, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta el pipeline completo de Tarea A y devuelve (comentarios_out, narrativas_out)."""
    c = df.copy()
    c["respuesta_a"] = c["respuesta_a"].fillna("")
    c["texto"] = c["texto"].fillna("")
    c["norm"] = c["texto"].map(normalizar)
    c["nwords"] = c["norm"].str.split().str.len()
    c["dt"] = pd.to_datetime(c["fecha"], utc=True)

    # Deduplicacion por texto normalizado
    u = c["norm"].drop_duplicates().sort_values().tolist()
    u2i = {s: i for i, s in enumerate(u)}
    c["u"] = c["norm"].map(u2i)

    # Filtro de ruido obvio antes de clustering (textos con >= 4 palabras y >= 12 chars)
    mask_u = np.array([len(s.split()) >= 4 and len(s) >= 12 for s in u])
    u_idx = np.where(mask_u)[0]
    u_filt = [u[i] for i in u_idx]

    E = calcular_embeddings(u_filt, cache_dir / "emb_filtered.npy")
    n = len(E)

    # Etapa 1: Micro-clusters ultra-puros en GPU k-NN + Leiden
    top_vals, top_idx = _knn_gpu(E, k=30)
    rows, cols = np.where(top_vals >= 0.76)
    g = ig.Graph(n=n, edges=list(zip(rows.tolist(), top_idx[rows, cols].tolist())), directed=False)
    g.es["weight"] = top_vals[rows, cols].tolist()
    g.simplify(combine_edges="max")
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=0.50, seed=42
    )
    lab = np.array(part.membership)
    vals, cnts = np.unique(lab, return_counts=True)
    min_micro = max(12, int(0.00025 * n))
    chicos = set(vals[cnts < min_micro])
    lab = np.array([-1 if l in chicos else l for l in lab])

    uniq_l = sorted(set(lab) - {-1})
    cents = {l: E[lab == l].mean(axis=0) for l in uniq_l}
    cents = {l: v / (np.linalg.norm(v) + 1e-9) for l, v in cents.items()}
    sim_cent = np.array([float(E[i] @ cents[lab[i]]) if lab[i] >= 0 else 0.0 for i in range(n)])
    lab = np.where(sim_cent >= 0.56, lab, -1)

    lab_all = np.full(len(u), -1, dtype=int)
    lab_all[u_idx] = lab
    c_micro = lab_all[c["u"].values]

    # Etapa 2: Grafo de micro-clusters por afinidad de respuesta_a + similitud coseno
    c_cid = c["comentario_id"].values
    c_resp = c["respuesta_a"].values
    cid_to_row = {cid: i for i, cid in enumerate(c_cid)}

    uniq_m = sorted(set(c_micro) - {-1})
    m2i = {m: i for i, m in enumerate(uniq_m)}
    M = len(uniq_m)
    R_mat = np.zeros((M, M), dtype=np.float64)
    for i in range(len(c)):
        a = c_micro[i]
        if a < 0 or c_resp[i] == "":
            continue
        p = cid_to_row.get(c_resp[i])
        if p is None:
            continue
        b = c_micro[p]
        if b < 0 or a == b:
            continue
        R_mat[m2i[a], m2i[b]] += 1
        R_mat[m2i[b], m2i[a]] += 1

    deg = R_mat.sum(axis=1, keepdims=True)
    expected = (deg @ deg.T) / (R_mat.sum() + 1e-9)
    lift = np.where((R_mat >= 3) & (expected > 0), R_mat / (expected + 1e-9), 0.0)
    C_mat = np.vstack([cents[m] for m in uniq_m])
    cos_mat = C_mat @ C_mat.T

    edges_m, wts_m = [], []
    for i in range(M):
        for j in range(i + 1, M):
            w = (min(np.log1p(lift[i, j]), 4.0) if lift[i, j] >= 2.0 else 0.0) + (
                2.0 * cos_mat[i, j] if cos_mat[i, j] >= 0.82 else 0.0
            )
            if w > 0:
                edges_m.append((i, j))
                wts_m.append(float(w))
    gm = ig.Graph(n=M, edges=edges_m, directed=False)
    gm.es["weight"] = wts_m
    pm = leidenalg.find_partition(
        gm, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=1.0, seed=42
    )
    lmap = {uniq_m[i]: pm.membership[i] for i in range(M)}

    # Etiquetas temporales de macro-clusters + herencia de respuestas
    c_macro = np.array([lmap[m] if m >= 0 else -1 for m in c_micro], dtype=int)
    for _ in range(3):
        changed = False
        for i in range(len(c)):
            if c_macro[i] < 0 and c_resp[i] != "":
                p = cid_to_row.get(c_resp[i])
                if p is not None and c_macro[p] >= 0:
                    c_macro[i] = c_macro[p]
                    changed = True
        if not changed:
            break

    # Renumerar macro-clusters de forma determinista (por tamano descendente: n01, n02, ...)
    # Filtrar clusters minusculos o spam sin debate (tasa de respuestas < 3%) como ruido
    counts = pd.Series(c_macro[c_macro >= 0]).value_counts()
    valid_macs = []
    for k, v in counts.items():
        if v < max(100, int(0.002 * len(c))):
            continue
        rep_rate = float((c_resp[c_macro == k] != "").mean())
        if rep_rate < 0.03:
            continue
        valid_macs.append(int(k))
    mac_to_lbl = {m: f"n{idx + 1:02d}" for idx, m in enumerate(valid_macs)}
    c["narrativa"] = [mac_to_lbl.get(m, "") for m in c_macro]
    c["micro"] = c_micro

    # c-TF-IDF por narrativa para frases clave y seleccion de representativos
    narr_labels = sorted(set(mac_to_lbl.values()))
    docs = [" ".join(c.loc[c["narrativa"] == lbl, "norm"].tolist()) for lbl in narr_labels]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.75, stop_words=STOPWORDS, sublinear_tf=True)
    X_tf = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())

    norms = c["norm"].values
    has_ref = np.array([any(m in s for m in REFUTE_PURE) for s in norms])
    has_agr = np.array([any(m in s for m in AGREE_PURE) for s in norms])
    c["ref"] = has_ref

    # Mapear cada texto normalizado en u_filt a su fila en E para calculo de centroides y LLM/postura
    u_filt_map = {s: i for i, s in enumerate(u_filt)}
    t_min_global = c["dt"].min()
    narr_rows = []
    proto_map = {}

    for idx_n, lbl in enumerate(narr_labels):
        sub = c[c["narrativa"] == lbl]
        # Top frases clave
        row_tf = X_tf[idx_n].toarray().ravel()
        top_t = [terms[j] for j in np.argsort(-row_tf)[:6]]

        # 15 textos representativos diversos (1 por micro-cluster principal, ordenados por frecuencia)
        rep_texts = []
        for m_id, _ in sub[sub["micro"] >= 0]["micro"].value_counts().head(15).items():
            rep_texts.append(sub[sub["micro"] == m_id]["texto"].value_counts().index[0])
        if len(rep_texts) < 5:
            rep_texts = sub.loc[sub["nwords"] >= 5, "texto"].value_counts().head(15).index.tolist()

        # Llamada opcional al LLM (1 por narrativa, cacheada)
        llm_res = _llamar_llm_cluster(rep_texts, cache_dir / "llm_labels.json", f"{lbl}_{len(sub)}")
        if llm_res:
            titulo = llm_res.get("titulo", rep_texts[0][:90])
            frases = "|".join(llm_res.get("frases_clave", top_t)[:6])
            proto_map[lbl] = (llm_res.get("apoya", []), llm_res.get("rechaza", []))
        else:
            # Fallback local: texto representativo afirmativo que contenga alguna frase clave del cluster
            cands = sub[
                (sub["nwords"] >= 7)
                & (~sub["ref"])
                & (~sub["texto"].str.strip().str.endswith("?"))
                & (~sub["texto"].str.contains("@", regex=False))
                & (~sub["norm"].str.startswith(("pasen ", "alguien ", "donde ", "cuando ", "como ")))
            ]
            top_terms_1 = [t.split()[0] for t in top_t[:4]]
            cands_key = cands[cands["norm"].apply(lambda s: any(tk in s for tk in top_terms_1))]
            if len(cands_key):
                titulo = cands_key["texto"].value_counts().index[0]
            elif len(cands):
                titulo = cands["texto"].value_counts().index[0]
            else:
                titulo = rep_texts[0]
            frases = "|".join(top_t)

        origen = _estimar_origen(sub["dt"], t_min_global).tz_convert("-03:00")
        narr_rows.append({
            "narrativa": lbl,
            "titulo": titulo,
            "frases_clave": frases,
            "fecha_origen": origen.isoformat(timespec="seconds"),
        })

    # Postura (apoya / rechaza)
    pred_st = np.full(len(c), "", dtype=object)
    if proto_map and len(proto_map) == len(narr_labels):
        # Con prototipos del LLM: clasificar cada micro-cluster comparando centroide vs prototipos apoya/rechaza
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(EMBED_MODEL, revision=EMBED_REV, device=device)
        for lbl in narr_labels:
            ap_txts, re_txts = proto_map[lbl]
            e_ap = model.encode([normalizar(x) for x in ap_txts], normalize_embeddings=True).mean(axis=0)
            e_re = model.encode([normalizar(x) for x in re_txts], normalize_embeddings=True).mean(axis=0)
            e_ap /= np.linalg.norm(e_ap) + 1e-9
            e_re /= np.linalg.norm(e_re) + 1e-9
            sub_idx = np.where((c["narrativa"].values == lbl))[0]
            for i in sub_idx:
                ui = u_filt_map.get(norms[i])
                if ui is not None:
                    pred_st[i] = "apoya" if (E[ui] @ e_ap) >= (E[ui] @ e_re) else "rechaza"
                else:
                    pred_st[i] = "apoya"
    else:
        micro_stance = _clasificar_postura_nli(c, narr_labels, force_cpu=False)
        if micro_stance is not None:
            for i in range(len(c)):
                if c["narrativa"].iat[i] == "":
                    continue
                m = c_micro[i]
                if m >= 0 and m in micro_stance:
                    pred_st[i] = micro_stance[m]
                elif has_ref[i]:
                    pred_st[i] = "rechaza"
        else:
            # Fallback sin NLI: marcadores de refutacion por micro-cluster
            micro_ref = c[c["micro"] >= 0].groupby("micro")["ref"].mean().to_dict()
            for i in range(len(c)):
                if c["narrativa"].iat[i] == "":
                    continue
                m = c_micro[i]
                if has_ref[i]:
                    pred_st[i] = "rechaza"
                elif m >= 0 and micro_ref.get(m, 0.0) >= 0.08:
                    pred_st[i] = "rechaza"
                else:
                    pred_st[i] = "apoya"

    # Herencia de postura en respuestas (regla estructural del dataset)
    for _ in range(3):
        for i in range(len(c)):
            if c["narrativa"].iat[i] == "":
                continue
            if c_resp[i] != "" and (pred_st[i] == "" or c["nwords"].iat[i] < 4):
                p = cid_to_row.get(c_resp[i])
                if p is not None and pred_st[p] != "":
                    if has_ref[i]:
                        pred_st[i] = "rechaza" if pred_st[p] == "apoya" else "apoya"
                    elif has_agr[i] or c["nwords"].iat[i] < 4:
                        pred_st[i] = pred_st[p]

    c["postura"] = np.where(c["narrativa"] == "", "", np.where(pred_st == "", "apoya", pred_st))
    return c[["comentario_id", "narrativa", "postura"]], pd.DataFrame(narr_rows)
