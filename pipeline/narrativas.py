"""Tarea A: Descubrimiento de narrativas, fecha de origen (changepoint), titulo/frases clave y postura."""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from pipeline.common import medir, normalizar, registrar_tokens

STOPWORDS = sorted({
    "a", "al", "algo", "con", "como", "de", "del", "el", "ella", "en", "es", "esa", "ese", "eso",
    "esta", "este", "esto", "la", "las", "le", "lo", "los", "mas", "me", "mi", "muy", "no", "nos",
    "o", "para", "pero", "por", "que", "se", "si", "sin", "su", "sus", "te", "todo", "todos", "un",
    "una", "uno", "y", "ya", "ahi", "aca", "hay", "ver", "digo", "ser", "son", "tan",
})

REFUTE_PURE = [
    "mentira", "mentiras", "falso", "falsa", "fake", "verso", "nada que ver", "no es cierto",
    "trucho", "trucha", "desmintio", "desmintieron", "sin pruebas", "inventan", "invento",
    "dejen de", "no es verdad", "puro cuento", "chamuyo", "humo", "opereta", "de otro lado",
    "no tiene sentido", "es viejo", "audio viejo", "foto vieja", "video viejo", "cualquiera",
]
AGREE_PURE = [
    "coincido", "tal cual", "exacto", "100%", "posta", "asi es", "clarisimo", "totalmente",
    "eso mismo", "ni mas ni menos", "lo mismo digo", "de acuerdo", "toda la razon", "confirmo",
]

# Regex compiladas con limites de palabra para evitar falsos positivos (ej: "verso" en "universo")
_REFUTE_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(w) for w in REFUTE_PURE) + r")(?!\w)"
)
_AGREE_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(w) for w in AGREE_PURE) + r")(?!\w)"
)

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
    k_eff = max(1, min(k, n - 1))
    device = "cpu" if (force_cpu or not torch.cuda.is_available()) else "cuda"
    dtype = torch.float16 if device == "cuda" else torch.float32
    Et = torch.from_numpy(E).to(device, dtype=dtype)
    top_vals = np.empty((n, k_eff), dtype=np.float32)
    top_idx = np.empty((n, k_eff), dtype=np.int32)
    bs = 2048 if device == "cuda" else 1024
    with torch.no_grad():
        for i in range(0, n, bs):
            sims = Et[i:i + bs] @ Et.T
            vals, inds = torch.topk(sims, k_eff + 1, dim=1)
            top_vals[i:i + bs] = vals[:, 1:].float().cpu().numpy()
            top_idx[i:i + bs] = inds[:, 1:].cpu().numpy()
    return top_vals, top_idx


def _estimar_origen(dts: pd.Series, t_min_global: pd.Timestamp) -> pd.Timestamp:
    """Changepoint robusto: detecta nacimiento a mitad de periodo ignorando goteo aislado previo."""
    s = dts.sort_values()
    if len(s) < 20:
        return s.quantile(0.05)
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


IA_CACHE_REPO = Path(__file__).resolve().parent / "ia_cache.json"


def _llamar_llm_cluster(ejemplos: list[str], cache_file: Path, key: str) -> dict | None:
    """Llama 1 vez por narrativa (temperature=0, seed=42) y registra respuesta en cache determinista."""
    def _valid_posts(p):
        if isinstance(p, dict) and len(p) >= max(5, len(ejemplos) // 2):
            return True
        if isinstance(p, list) and len(p) >= max(5, len(ejemplos) // 2):
            return True
        return False

    for cf in (cache_file, IA_CACHE_REPO):
        if cf.exists():
            try:
                cache_d = json.loads(cf.read_text(encoding="utf-8"))
                if key in cache_d and _valid_posts(cache_d[key].get("posturas")):
                    entry = cache_d[key]
                    registrar_tokens(entry.get("_tokens", 0), entry.get("_usd", 0.0))
                    return entry
            except Exception:
                pass

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(ejemplos))
    n_ej = len(ejemplos)
    prompt = (
        "Sos un analista experto en detección de narrativas en redes sociales argentinas.\n"
        f"Abajo hay {n_ej} comentarios representativos (numerados de 0 a {n_ej - 1}) que pertenecen a un mismo clúster temático.\n\n"
        "REGLAS ESTRICTAS DE ANÁLISIS:\n"
        "1. ANCLAJE DE LA TESIS CENTRAL ('titulo'): Identificá el reclamo, acusación, rumor o afirmación originaria que dispara la discusión. "
        "Redactá el 'titulo' SIEMPRE como una afirmación directa y activa (ej: 'La obra se frenó porque desviaron los fondos'), "
        "NUNCA como un tema neutro ('Debate sobre la obra') ni como una negación/desmentida, para no invertir las posturas.\n"
        "2. VARIACIÓN LÉXICA ('frases_clave'): Extraé exactamente 5 frases cortas o sintagmas representativos que capturen las distintas formas "
        "en que los usuarios expresan esta narrativa.\n"
        "3. CLASIFICACIÓN DE POSTURA ('posturas'): Evaluá cada comentario (0 a "
        f"{n_ej - 1}) respecto a la afirmación exacta del 'titulo', detectando sarcasmo e ironía argentina:\n"
        '   - "apoya": afirma, amplifica, da por cierta o ironiza a favor de la tesis del "titulo".\n'
        '   - "rechaza": niega, desmiente, defiende a la parte acusada, minimiza o refuta la tesis del "titulo".\n'
        "   (Nota: No asumas una proporción fija; clasificá cada comentario por su intención real).\n\n"
        "FORMATO DE SALIDA:\n"
        "Respondé ÚNICAMENTE con un objeto JSON crudo y válido (sin bloques ```json ni texto adicional) que respete este esquema exacto:\n"
        "{\n"
        '  "titulo": "Afirmación central corta y directa",\n'
        '  "frases_clave": ["frase 1", "frase 2", "frase 3", "frase 4", "frase 5"],\n'
        f'  "posturas": {{"0": "apoya", "1": "rechaza", "{n_ej - 1}": "apoya"}}\n'
        "}\n\n"
        f"Comentarios:\n{numbered}"
    )

    def _save_entry(out: dict, toks: int, usd: float) -> dict:
        out["_tokens"] = int(toks)
        out["_usd"] = round(float(usd), 6)
        registrar_tokens(toks, usd)
        try:
            data_c = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
            data_c[key] = out
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data_c, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return out

    try:
        import urllib.error
        import urllib.request
        if os.environ.get("GEMINI_API_KEY"):
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.5-flash:generateContent?key={os.environ['GEMINI_API_KEY']}"
            )
            body = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "seed": 42,
                    "responseMimeType": "application/json",
                    "thinkingConfig": {"thinkingBudget": 256},
                },
            }
            for intento in range(4):
                try:
                    if intento > 0:
                        time.sleep(2.0)
                    req = urllib.request.Request(
                        url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        meta = data.get("usageMetadata", {})
                        p_tok = int(meta.get("promptTokenCount", 0))
                        o_tok = int(meta.get("candidatesTokenCount", 0)) + int(meta.get("thoughtsTokenCount", 0))
                        toks = int(meta.get("totalTokenCount", p_tok + o_tok))
                        usd = (p_tok * 0.30 + o_tok * 2.50) / 1e6
                        txt = data["candidates"][0]["content"]["parts"][0]["text"]
                        out = json.loads(txt)
                        if _valid_posts(out.get("posturas")):
                            return _save_entry(out, toks, usd)
                        break
                except urllib.error.HTTPError as e:
                    err_txt = ""
                    try:
                        err_txt = e.read().decode("utf-8")[:180]
                    except Exception:
                        pass
                    if e.code in (429, 503) and intento < 3:
                        time.sleep(15 * (intento + 1))
                        continue
                    print(f"[LLM warn] HTTP {e.code} en key={key}: {err_txt}", flush=True)
                    raise
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
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                toks = int(data.get("usage", {}).get("total_tokens", 0))
                usd = toks * 0.0000003
                out = json.loads(data["choices"][0]["message"]["content"])
                if _valid_posts(out.get("posturas")):
                    return _save_entry(out, toks, usd)
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
        if device == "cuda":
            nli.half()
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
            probs = torch.softmax(nli(**inp).logits.float(), dim=-1).cpu().numpy()
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
    c_cid = c["comentario_id"].to_numpy()
    c_resp = c["respuesta_a"].to_numpy()
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

    # Deteccion lexica con word boundaries sobre textos unicos (mucho mas rapido y sin falsos positivos)
    u_ref = np.array([bool(_REFUTE_RE.search(s)) for s in u], dtype=bool)
    u_agr = np.array([bool(_AGREE_RE.search(s)) for s in u], dtype=bool)
    u_vals = c["u"].to_numpy()
    has_ref = u_ref[u_vals]
    has_agr = u_agr[u_vals]
    c["ref"] = has_ref

    u_filt_map = {s: i for i, s in enumerate(u_filt)}
    t_min_global = c["dt"].min()
    narr_rows = []
    ia_samples_map = {}

    for idx_n, lbl in enumerate(narr_labels):
        sub = c[c["narrativa"] == lbl]
        row_tf = X_tf[idx_n].toarray().ravel()
        top_t = [terms[j] for j in np.argsort(-row_tf)[:6]]

        # Muestra diversa de hasta 60 comentarios (1 por micro-cluster principal + textos mas frecuentes)
        rep_texts = []
        seen_norms = set()
        for m_id, _ in sub[sub["micro"] >= 0]["micro"].value_counts().head(35).items():
            sub_m = sub[(sub["micro"] == m_id) & (sub["nwords"] >= 4)]
            if len(sub_m) == 0:
                sub_m = sub[sub["micro"] == m_id]
            tx = sub_m["texto"].value_counts().index[0]
            nm = normalizar(tx)
            if nm not in seen_norms:
                seen_norms.add(nm)
                rep_texts.append(tx)
        freq_norms = (
            sub[sub["nwords"] >= 4]
            .groupby("norm", as_index=False)
            .agg(cnt=("texto", "count"), tx=("texto", "first"))
            .sort_values(["cnt", "norm"], ascending=[False, True])
        )
        for row_f in freq_norms.itertuples():
            if len(rep_texts) >= 60:
                break
            if row_f.norm not in seen_norms:
                seen_norms.add(row_f.norm)
                rep_texts.append(row_f.tx)
        if len(rep_texts) < 5:
            rep_texts = sub["texto"].value_counts().head(15).index.tolist()

        # Llamada 1 vez por narrativa (cacheada por hash de la muestra)
        sample_key = hashlib.sha256("\n".join(rep_texts).encode("utf-8")).hexdigest()[:16]
        llm_res = _llamar_llm_cluster(rep_texts, cache_dir / "llm_labels.json", sample_key)
        if llm_res:
            titulo = llm_res.get("titulo", rep_texts[0][:90])
            frases = "|".join(llm_res.get("frases_clave", top_t)[:6])
            ia_samples_map[lbl] = (rep_texts, llm_res["posturas"])
        else:
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
    norms = c["norm"].to_numpy()
    c_narr = c["narrativa"].to_numpy()
    c_nwords = c["nwords"].to_numpy()

    if ia_samples_map:
        dim_e = E.shape[1]
        zeros_e = np.zeros(dim_e, dtype=np.float32)

        for lbl, (s_txts, s_posts) in ia_samples_map.items():
            lbl_map: dict[str, int] = {}
            pairs_it = (
                ((s_txts[int(k)], v) for k, v in s_posts.items() if str(k).isdigit() and int(k) < len(s_txts))
                if isinstance(s_posts, dict)
                else zip(s_txts, s_posts)
            )
            for tx, p_val in pairs_it:
                p_clean = str(p_val).strip().lower()
                if p_clean in ("apoya", "rechaza"):
                    lbl_map[normalizar(tx)] = 1 if p_clean == "apoya" else 0

            sub_idx = np.where(c_narr == lbl)[0]
            sub_norms = [norms[i] for i in sub_idx]
            uniq_sub_norms = sorted(set(sub_norms))
            norm_to_u_idx = {nm: idx for idx, nm in enumerate(uniq_sub_norms)}

            tr_u = [idx for idx, nm in enumerate(uniq_sub_norms) if nm in lbl_map]
            y_tr_u = np.array([lbl_map[uniq_sub_norms[idx]] for idx in tr_u], dtype=int)

            if len(set(y_tr_u)) < 2:
                val = "apoya" if (len(y_tr_u) == 0 or y_tr_u[0] == 1) else "rechaza"
                pred_st[sub_idx] = val
            else:
                # Entrenamos y predecimos sobre textos unicos del cluster (sin sesgo por spam repetido)
                vec_st = TfidfVectorizer(ngram_range=(1, 2), max_features=2500, sublinear_tf=True)
                X_tf_u = vec_st.fit_transform(uniq_sub_norms)
                X_emb_u = np.vstack([E[u_filt_map[nm]] if nm in u_filt_map else zeros_e for nm in uniq_sub_norms])
                X_lex_u = np.column_stack([
                    [1.0 if _REFUTE_RE.search(nm) else 0.0 for nm in uniq_sub_norms],
                    [1.0 if _AGREE_RE.search(nm) else 0.0 for nm in uniq_sub_norms],
                ]).astype(np.float32)
                X_u = hstack([X_tf_u, csr_matrix(X_emb_u), csr_matrix(X_lex_u * 0.6)]).tocsr()

                clf = LogisticRegression(C=2.0, max_iter=500, class_weight="balanced", random_state=42)
                clf.fit(X_u[tr_u], y_tr_u)
                probs_u = clf.predict_proba(X_u)[:, 1]

                # Suavizado por micro-cluster: los miembros de un mismo micro-cluster comparten postura
                row_probs = np.array([probs_u[norm_to_u_idx[nm]] for nm in sub_norms], dtype=np.float32)
                sub_micros = c_micro[sub_idx]
                micro_prob_sum: dict[int, float] = {}
                micro_prob_cnt: dict[int, int] = {}
                for li, m_id in enumerate(sub_micros):
                    if m_id >= 0:
                        nm = sub_norms[li]
                        p_val = float(lbl_map[nm]) if nm in lbl_map else float(row_probs[li])
                        micro_prob_sum[m_id] = micro_prob_sum.get(m_id, 0.0) + p_val
                        micro_prob_cnt[m_id] = micro_prob_cnt.get(m_id, 0) + 1

                for li, gi in enumerate(sub_idx):
                    nm = sub_norms[li]
                    if nm in lbl_map:
                        pred_st[gi] = "apoya" if lbl_map[nm] == 1 else "rechaza"
                    else:
                        m_id = sub_micros[li]
                        if m_id >= 0 and micro_prob_cnt.get(m_id, 0) >= 3:
                            m_mean = micro_prob_sum[m_id] / micro_prob_cnt[m_id]
                            final_p = 0.60 * row_probs[li] + 0.40 * m_mean
                        else:
                            final_p = row_probs[li]
                        pred_st[gi] = "apoya" if final_p >= 0.5 else "rechaza"

    missing_lbls = [lbl for lbl in narr_labels if lbl not in ia_samples_map]
    if missing_lbls:
        micro_stance = _clasificar_postura_nli(c, missing_lbls, force_cpu=False)
        miss_set = set(missing_lbls)
        if micro_stance is not None:
            for i in range(len(c)):
                if c_narr[i] not in miss_set:
                    continue
                m = c_micro[i]
                if m >= 0 and m in micro_stance:
                    pred_st[i] = micro_stance[m]
                elif has_ref[i]:
                    pred_st[i] = "rechaza"
        else:
            micro_ref = c[c["micro"] >= 0].groupby("micro")["ref"].mean().to_dict()
            for i in range(len(c)):
                if c_narr[i] not in miss_set:
                    continue
                m = c_micro[i]
                if has_ref[i]:
                    pred_st[i] = "rechaza"
                elif m >= 0 and micro_ref.get(m, 0.0) >= 0.08:
                    pred_st[i] = "rechaza"
                else:
                    pred_st[i] = "apoya"

    # Herencia de postura en respuestas (vectorizada sobre arrays de NumPy)
    for _ in range(3):
        for i in range(len(c)):
            if c_narr[i] == "":
                continue
            if c_resp[i] != "" and (pred_st[i] == "" or c_nwords[i] < 4 or has_agr[i]):
                p = cid_to_row.get(c_resp[i])
                if p is not None and pred_st[p] != "":
                    if has_ref[i]:
                        pred_st[i] = "rechaza" if pred_st[p] == "apoya" else "apoya"
                    elif has_agr[i] or c_nwords[i] < 4:
                        pred_st[i] = pred_st[p]

    c["postura"] = np.where(c_narr == "", "", np.where(pred_st == "", "apoya", pred_st))
    return c[["comentario_id", "narrativa", "postura"]], pd.DataFrame(narr_rows)
