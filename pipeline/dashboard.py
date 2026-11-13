"""Genera docs/dashboard.html: reporte interactivo autocontenido (linea de tiempo, posturas, grafo y fichas CIB)."""
import json
import re
from pathlib import Path

import pandas as pd


def generar_dashboard(
    df_raw: pd.DataFrame,
    df_com: pd.DataFrame,
    df_narr: pd.DataFrame,
    df_cta: pd.DataFrame,
    aristas: list[dict],
    costo: dict,
    out_path: Path,
):
    m = df_raw[["comentario_id", "fecha"]].merge(df_com, on="comentario_id")
    m["dia"] = pd.to_datetime(m["fecha"], utc=True).dt.tz_convert("-03:00").dt.strftime("%Y-%m-%d")
    valid = m[m["narrativa"] != ""]

    # Serie temporal diaria por narrativa
    dias = sorted(valid["dia"].unique().tolist())
    series = []
    for _, row in df_narr.iterrows():
        nid = row["narrativa"]
        sub = valid[valid["narrativa"] == nid]
        counts = sub.groupby("dia").size().to_dict()
        n_ap = int((sub["postura"] == "apoya").sum())
        n_re = int((sub["postura"] == "rechaza").sum())
        series.append({
            "id": nid,
            "titulo": row["titulo"][:85],
            "frases": row["frases_clave"],
            "origen": row["fecha_origen"][:16].replace("T", " "),
            "total": len(sub),
            "apoya": n_ap,
            "rechaza": n_re,
            "diario": [int(counts.get(d, 0)) for d in dias],
        })
    series.sort(key=lambda x: -x["total"])

    # Resumen y fichas por operacion coordinada
    coord_df = df_cta[df_cta["grupo"] != ""]
    grupos_resumen = coord_df.groupby("grupo").size().sort_values(ascending=False).to_dict()
    nodos = [{"id": r.cuenta_id, "grupo": r.grupo} for r in coord_df.itertuples()]

    fichas_cib = []
    if len(coord_df):
        c_coord = df_raw.merge(coord_df, on="cuenta_id")
        c_coord["dt_ar"] = pd.to_datetime(c_coord["fecha"], utc=True).dt.tz_convert("-03:00")
        for g_id, n_ctas in grupos_resumen.items():
            sub_g = c_coord[c_coord["grupo"] == g_id]
            tags = []
            for tx in sub_g["texto"]:
                if "#" in tx:
                    tags.extend(re.findall(r"#[a-z0-9_ñáéíóú]+", tx.lower()))
            top_tags = [k for k, _ in pd.Series(tags).value_counts().head(4).items()] if tags else []
            top_pubs = sub_g["publicacion_id"].value_counts().head(3).index.tolist()
            n_ed = sum(1 for e in aristas if e.get("grupo") == g_id)
            ej = sub_g["texto"].value_counts().head(2).index.tolist()
            fichas_cib.append({
                "grupo": g_id,
                "cuentas": int(n_ctas),
                "comentarios": int(len(sub_g)),
                "aristas": int(n_ed),
                "hashtags": top_tags,
                "pubs": top_pubs,
                "desde": sub_g["dt_ar"].min().strftime("%Y-%m-%d %H:%M"),
                "hasta": sub_g["dt_ar"].max().strftime("%Y-%m-%d %H:%M"),
                "ejemplos": [x[:95] for x in ej],
            })

    payload = {
        "costo": costo,
        "dias": dias,
        "narrativas": series,
        "ruido": int((m["narrativa"] == "").sum()),
        "grupos": grupos_resumen,
        "fichas_cib": fichas_cib,
        "nodos": nodos,
        "aristas": aristas,
    }

    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>QualyData — Reporte de Narrativas y Coordinacion</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
  h1, h2 {{ margin: 0 0 12px; color: #f8fafc; }}
  .sub {{ color: #94a3b8; margin-bottom: 20px; font-size: 14px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 24px; }}
  .kpi {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; }}
  .kpi .v {{ font-size: 24px; font-weight: 700; color: #38bdf8; }}
  .kpi .l {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
  .grid {{ display: grid; grid-template-columns: 1.3fr 1fr; gap: 16px; margin-bottom: 20px; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px 6px; border-bottom: 1px solid #334155; }}
  th {{ color: #94a3b8; font-weight: 600; }}
  .bar {{ display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: #334155; min-width: 90px; }}
  .ap {{ background: #22c55e; }}
  .re {{ background: #ef4444; }}
  .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; background: #0ea5e933; color: #38bdf8; }}
  canvas {{ width: 100%; height: 280px; display: block; }}
</style>
</head>
<body>
  <h1>QualyData — Tablero Ejecutivo de Narrativas y CIB</h1>
  <div class="sub">Motor: {costo.get('almacenamiento','DuckDB + Parquet')} · Hardware: {costo.get('hardware','')}</div>
  <div class="kpis" id="kpis"></div>
  <div class="grid">
    <div class="card">
      <h2>Aparicion y Evolucion de Narrativas</h2>
      <canvas id="timeline" width="760" height="290"></canvas>
    </div>
    <div class="card">
      <h2>Red de Operaciones Coordinadas (CIB)</h2>
      <canvas id="net" width="520" height="290"></canvas>
      <div id="grupos-info" style="font-size:12px;color:#94a3b8;margin-top:8px;"></div>
    </div>
  </div>
  <div class="card" style="margin-bottom:20px;">
    <h2>Fichas por Operacion Coordinada (CIB)</h2>
    <div id="fichas" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;"></div>
  </div>
  <div class="card">
    <h2>Narrativas Descubiertas, Origen y Postura (Apoya vs Rechaza)</h2>
    <table>
      <thead><tr><th>ID</th><th>Titulo / Tesis</th><th>Origen (-03:00)</th><th>Volumen</th><th>Postura (Apoya / Rechaza)</th><th>Frases clave</th></tr></thead>
      <tbody id="tbl"></tbody>
    </table>
  </div>
<script>
const D = {json.dumps(payload, ensure_ascii=False)};
const kpis = [
  ['Comentarios', D.costo.comentarios_procesados.toLocaleString()],
  ['Narrativas', D.narrativas.length],
  ['Cuentas Coordinadas', D.nodos.length + ' (' + Object.keys(D.grupos).length + ' ops)'],
  ['Tiempo Total', D.costo.segundos + ' s'],
  ['Costo API', '$' + D.costo.usd_ia.toFixed(4) + ' USD']
];
document.getElementById('kpis').innerHTML = kpis.map(([l,v]) => `<div class="kpi"><div class="v">${{v}}</div><div class="l">${{l}}</div></div>`).join('');

const PAL = ['#38bdf8','#22c55e','#f59e0b','#ef4444','#a855f7','#ec4899','#14b8a6','#eab308','#6366f1','#fb7185'];
document.getElementById('fichas').innerHTML = D.fichas_cib.map((f,i) => `
  <div style="background:#0f172a;border:1px solid #334155;border-radius:8px;padding:12px;font-size:12px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
      <span class="pill" style="color:${{PAL[i%PAL.length]}};font-weight:700;">${{f.grupo}}</span>
      <span style="color:#94a3b8;">${{f.cuentas}} cuentas · ${{f.comentarios.toLocaleString()}} com.</span>
    </div>
    <div style="color:#cbd5e1;margin-bottom:4px;"><b>Actividad:</b> ${{f.desde}} a ${{f.hasta}}</div>
    <div style="color:#cbd5e1;margin-bottom:4px;"><b>Hashtags:</b> ${{f.hashtags.length ? f.hashtags.join(', ') : 'Sin hashtags (co-sincronía)'}}</div>
    <div style="color:#cbd5e1;margin-bottom:4px;"><b>Foco pubs:</b> ${{f.pubs.join(', ')}}</div>
    <div style="color:#94a3b8;font-style:italic;">"${{f.ejemplos[0] || ''}}"</div>
  </div>
`).join('') || '<div style="color:#94a3b8;font-size:13px;">Sin operaciones detectadas</div>';
document.getElementById('tbl').innerHTML = D.narrativas.map((n,i) => {{
  const pAp = (100 * n.apoya / Math.max(n.total,1)).toFixed(0);
  const pRe = 100 - pAp;
  return `<tr>
    <td><span class="pill" style="color:${{PAL[i%PAL.length]}}">${{n.id}}</span></td>
    <td>${{n.titulo}}</td>
    <td>${{n.origen}}</td>
    <td>${{n.total.toLocaleString()}}</td>
    <td><div class="bar"><div class="ap" style="width:${{pAp}}%"></div><div class="re" style="width:${{pRe}}%"></div></div><small>${{pAp}}% / ${{pRe}}%</small></td>
    <td style="color:#94a3b8">${{n.frases.split('|').slice(0,4).join(', ')}}</td>
  </tr>`;
}}).join('');

// Timeline chart
const tc = document.getElementById('timeline').getContext('2d');
const W = 760, H = 290, padL = 45, padB = 28, padT = 15, padR = 15;
const maxY = Math.max(1, ...D.narrativas.slice(0,8).flatMap(n => n.diario));
tc.strokeStyle = '#334155'; tc.lineWidth = 1;
for(let g=0; g<=4; g++) {{
  const y = padT + (H-padT-padB)*(1 - g/4);
  tc.beginPath(); tc.moveTo(padL, y); tc.lineTo(W-padR, y); tc.stroke();
  tc.fillStyle = '#64748b'; tc.font = '11px sans-serif';
  tc.fillText(Math.round(maxY*g/4), 4, y+4);
}}
D.dias.forEach((d, idx) => {{
  if (idx % 2 === 0) {{
    const x = padL + (W-padL-padR)*(idx/Math.max(D.dias.length-1,1));
    tc.fillStyle = '#94a3b8'; tc.fillText(d.slice(5), x-14, H-8);
  }}
}});
D.narrativas.slice(0,8).forEach((n, i) => {{
  tc.strokeStyle = PAL[i % PAL.length]; tc.lineWidth = 2; tc.beginPath();
  n.diario.forEach((v, idx) => {{
    const x = padL + (W-padL-padR)*(idx/Math.max(D.dias.length-1,1));
    const y = padT + (H-padT-padB)*(1 - v/maxY);
    if(idx===0) tc.moveTo(x,y); else tc.lineTo(x,y);
  }});
  tc.stroke();
}});

// Force-directed-like circular clusters for CIB network
const nc = document.getElementById('net').getContext('2d');
const gNames = Object.keys(D.grupos);
document.getElementById('grupos-info').innerText = gNames.map((g,i)=>`${{g}}: ${{D.grupos[g]}} cuentas`).join('  |  ') || 'Sin operaciones coordinadas';
const pos = {{}};
gNames.forEach((g, gi) => {{
  const cx = (520 / (gNames.length + 1)) * (gi + 1);
  const cy = 145;
  const members = D.nodos.filter(n => n.grupo === g);
  const rad = Math.min(75, 22 + members.length * 1.1);
  members.forEach((m, mi) => {{
    const ang = (2 * Math.PI * mi) / Math.max(members.length, 1);
    pos[m.id] = {{ x: cx + rad * Math.cos(ang), y: cy + rad * Math.sin(ang), c: PAL[gi % PAL.length] }};
  }});
}});
nc.strokeStyle = 'rgba(148,163,184,0.18)'; nc.lineWidth = 1;
D.aristas.forEach(e => {{
  const a = pos[e.source], b = pos[e.target];
  if(a && b) {{ nc.beginPath(); nc.moveTo(a.x, a.y); nc.lineTo(b.x, b.y); nc.stroke(); }}
}});
Object.values(pos).forEach(p => {{
  nc.fillStyle = p.c; nc.beginPath(); nc.arc(p.x, p.y, 4, 0, 2*Math.PI); nc.fill();
}});
</script>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")
