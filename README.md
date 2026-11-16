# Detección de narrativas y coordinación (QualyData · Big Data)

Proyecto para el desafío QualyData de la Competencia Argentina Space 2026. Toma cientos de miles de comentarios de redes sociales y responde tres preguntas: qué narrativas circulan, desde cuándo y qué cuentas actúan coordinadas para instalarlas. Corre con un solo comando y es determinista (`SEED = 42`).

## Resultado

| Métrica | Valor |
|---|---|
| Puntaje automático, set de práctica (evaluador oficial `evaluacion/puntuar.py`) | **53,57 / 55** (línea base oficial: ~17 / 55) |
| Mismo set, 100 % local, sin llamadas a LLM | 51,31 / 55 |
| Narrativas: agrupamiento (B-cubed F1) | 0,960 |
| Postura apoya/rechaza (macro-F1) | 0,974 (0,597 solo con NLI local) |
| Cuentas coordinadas (F1) | 0,987 |
| Set final: 150.000 comentarios, 12.002 cuentas | 22,96 s · US$ 0,013 en IA · 2,26 GB de RAM pico |

El puntaje es autoevaluado con el script oficial sobre el set de práctica, que trae la respuesta correcta. Los 45 puntos que pone el jurado (costo y escala, robustez y claridad) no están incluidos.

## Cómo funciona

1. **Ingesta:** pasa los `.csv.gz` a Parquet (zstd) y DuckDB local, y los lee con Polars. No hace falta ningún servidor.
2. **Narrativas:**
   - Deduplica los textos y descarta el ruido.
   - Genera embeddings locales con `paraphrase-multilingual-mpnet-base-v2`.
   - Agrupa en dos etapas: un grafo k-NN con Leiden y después une los micro-clústeres según quién le responde a quién.
   - Estima la fecha de origen de cada narrativa detectando el punto de quiebre en una grilla horaria.
3. **Postura:**
   - Toma una muestra por narrativa y la etiqueta una sola vez con `gemini-2.5-flash` (temperatura 0). Las respuestas quedan guardadas por hash SHA-256 en `pipeline/ia_cache.json`, así que un clon sin API key reproduce el resultado.
   - Si una narrativa no está en caché, usa el modelo NLI local `mDeBERTa-v3`.
   - Con esa muestra entrena una regresión logística sobre TF-IDF más embeddings.
4. **Coordinación:**
   - En DuckDB arma un grafo de cuentas que coinciden en ventanas de 30 minutos, dándole menos peso a las coincidencias en publicaciones virales.
   - Suma otras señales: textos casi idénticos, hashtags raros en común, respuestas mutuas y cuentas recién creadas.
   - Después aplica Leiden y se queda solo con las comunidades densas.
5. **Salida:** archivos CSV en `entrega/` y un tablero estático en `docs/dashboard.html`.

## Cómo correrlo

Requiere Python 3.10 o superior. Anda en GPU o solo con CPU, y no necesita API key para reproducir la entrega.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Set final
PYTHONHASHSEED=42 python correr.py --datos datos/final --entrega entrega --cache salida/cache_final

# Set de práctica + puntaje
PYTHONHASHSEED=42 python correr.py --datos datos/practica --entrega salida/practica --cache salida/cache
python evaluacion/puntuar.py --verdad datos/practica/verdad --entrega salida/practica
```

## Pendiente

- Tests automáticos y CI.
