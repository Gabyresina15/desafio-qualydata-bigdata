"""Tests de humo sin GPU, sin descargas de modelos ni API keys."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluacion"))

from pipeline.common import ingerir, normalizar
import puntuar


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


class TestSmokePipeline(unittest.TestCase):
    def test_1_ingesta_duckdb_parquet(self):
        """ingerir() convierte CSVs chicos a Parquet/DuckDB conservando la cantidad de filas."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            datos = tmp_p / "datos"
            db_dir = tmp_p / "db"
            datos.mkdir()

            n = 6
            _write_csv(
                datos / "comentarios.csv",
                ["comentario_id", "publicacion_id", "cuenta_id", "fecha", "texto", "respuesta_a", "me_gusta"],
                [
                    {
                        "comentario_id": f"c{i}",
                        "publicacion_id": "p1",
                        "cuenta_id": f"u{i}",
                        "fecha": f"2026-10-05T10:0{i}:00-03:00",
                        "texto": f"comentario de prueba numero {i}",
                        "respuesta_a": "",
                        "me_gusta": str(i),
                    }
                    for i in range(n)
                ],
            )
            _write_csv(
                datos / "cuentas.csv",
                ["cuenta_id", "usuario", "fecha_creacion", "seguidores", "siguiendo"],
                [
                    {
                        "cuenta_id": f"u{i}",
                        "usuario": f"user_{i}",
                        "fecha_creacion": "2025-01-01T00:00:00-03:00",
                        "seguidores": "10",
                        "siguiendo": "5",
                    }
                    for i in range(n)
                ],
            )
            _write_csv(
                datos / "publicaciones.csv",
                ["publicacion_id", "fuente_id", "fecha", "titular"],
                [
                    {
                        "publicacion_id": "p1",
                        "fuente_id": "f1",
                        "fecha": "2026-10-05T09:00:00-03:00",
                        "titular": "Titular de prueba",
                    }
                ],
            )

            con = ingerir(datos, db_dir)
            try:
                for tabla, esperado in (("comentarios", n), ("cuentas", n), ("publicaciones", 1)):
                    self.assertTrue((db_dir / f"{tabla}.parquet").exists())
                    cnt = con.execute(f"SELECT COUNT(*) FROM {tabla}").fetchone()[0]
                    self.assertEqual(cnt, esperado)
            finally:
                con.close()

    def test_2_puntuar_puntaje_perfecto(self):
        """evaluacion/puntuar.py da 30/30 + 25/25 cuando entrega coincide con verdad."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            verdad = tmp_p / "verdad"
            entrega = tmp_p / "entrega"
            verdad.mkdir()
            entrega.mkdir()

            _write_csv(
                verdad / "comentarios.csv",
                ["comentario_id", "narrativa_id", "postura"],
                [
                    {"comentario_id": "c1", "narrativa_id": "N1", "postura": "apoya"},
                    {"comentario_id": "c2", "narrativa_id": "N1", "postura": "rechaza"},
                    {"comentario_id": "c3", "narrativa_id": "ninguna", "postura": ""},
                ],
            )
            _write_csv(
                verdad / "narrativas.csv",
                ["narrativa_id", "emergente", "fecha_origen"],
                [{"narrativa_id": "N1", "emergente": "si", "fecha_origen": "2026-10-05T10:00:00-03:00"}],
            )
            _write_csv(
                verdad / "cuentas.csv",
                ["cuenta_id", "campana_id"],
                [{"cuenta_id": "u1", "campana_id": "C1"}, {"cuenta_id": "u2", "campana_id": ""}],
            )

            _write_csv(
                entrega / "comentarios.csv",
                ["comentario_id", "narrativa", "postura"],
                [
                    {"comentario_id": "c1", "narrativa": "n01", "postura": "apoya"},
                    {"comentario_id": "c2", "narrativa": "n01", "postura": "rechaza"},
                    {"comentario_id": "c3", "narrativa": "", "postura": ""},
                ],
            )
            _write_csv(
                entrega / "narrativas.csv",
                ["narrativa", "titulo", "frases_clave", "fecha_origen"],
                [
                    {
                        "narrativa": "n01",
                        "titulo": "Tesis",
                        "frases_clave": "a|b",
                        "fecha_origen": "2026-10-05T10:30:00-03:00",
                    }
                ],
            )
            _write_csv(
                entrega / "cuentas.csv",
                ["cuenta_id", "grupo"],
                [{"cuenta_id": "u1", "grupo": "g001"}, {"cuenta_id": "u2", "grupo": ""}],
            )

            narr = puntuar.score_narratives(verdad, entrega)
            coord = puntuar.score_coordination(verdad, entrega)
            self.assertEqual(narr["puntos"], 30.0)
            self.assertEqual(coord["puntos"], 25.0)

    def test_3_filtro_ruido_narrativas(self):
        """El filtro de ruido descarta textos de < 4 palabras y conserva textos largos."""
        textos = [
            "Tal cual!!",
            "Jajaja",
            "La obra del hospital sigue parada desde hace meses sin explicaciones.",
        ]
        norms = [normalizar(t) for t in textos]
        pasa = [len(s.split()) >= 4 and len(s) >= 12 for s in norms]
        self.assertEqual(pasa, [False, False, True])


if __name__ == "__main__":
    unittest.main()
