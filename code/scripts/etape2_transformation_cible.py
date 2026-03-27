#!/usr/bin/env python3
"""
Étape 2 du plan (code/guide/plan.md) : transformation de la variable cible pour le MLP.

- ε = 10⁻⁶ × min(D > 0)
- f = ln(D + ε)  (log naturel, usage standard en régression)
- (Optionnel) f_scaled = (f − μ_f) / σ_f

Entrée : kernel_propre.csv (sortie étape 1)
Sorties :
  - kernel_cible_log.csv : colonnes d'entraînement + métadonnées utiles
  - etape2_rapport.md : ε, statistiques de D vs f, chemins des figures
  - etape2_histogrammes.png : histogrammes D (log-x) vs f
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EPSILON_FACTOR = 1e-6


def _read_kernel_propre(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("R_norm", "cos_theta", "D"):
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans {path}: {col}")
    if (df["D"] <= 0).any():
        raise ValueError("Étape 2 attend D > 0 partout (exécuter l'étape 1 d'abord).")
    return df


def transform_target(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d_min = float(df["D"].min())
    epsilon = EPSILON_FACTOR * d_min
    f = np.log(df["D"].to_numpy(dtype=np.float64) + epsilon)
    mu_f = float(np.mean(f))
    sigma_f = float(np.std(f))
    if sigma_f <= 0:
        raise ValueError("σ_f nul : jeu dégénéré.")
    f_scaled = (f - mu_f) / sigma_f

    out = df.copy()
    out["epsilon"] = epsilon
    out["f"] = f
    out["f_scaled"] = f_scaled

    stats: dict = {
        "epsilon": epsilon,
        "D_min": d_min,
        "D_max": float(df["D"].max()),
        "mu_f": mu_f,
        "sigma_f": sigma_f,
        "f_min": float(f.min()),
        "f_max": float(f.max()),
        "skew_D": float(pd.Series(df["D"]).skew()),
        "skew_f": float(pd.Series(f).skew()),
    }
    return out, stats


def _plot_histograms(df: pd.DataFrame, f: np.ndarray, path_png: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    d = df["D"].to_numpy()
    axes[0].hist(np.log10(np.maximum(d, 1e-300)), bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("log₁₀(D)")
    axes[0].set_ylabel("Effectif")
    axes[0].set_title("D (échelle log₁₀)")

    axes[1].hist(f, bins=50, color="darkseagreen", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("f = ln(D + ε)")
    axes[1].set_ylabel("Effectif")
    axes[1].set_title("Cible transformée f")

    fig.suptitle("Étape 2 — comparaison des distributions", fontsize=11)
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=150)
    plt.close(fig)


def write_report(path: Path, stats: dict, paths: dict[str, str]) -> None:
    text = f"""# Rapport — Étape 2 (transformation de la cible)

## Paramètres

- **ε** = {EPSILON_FACTOR:g} × min(D) = **{stats['epsilon']:.6e}** (mêmes unités que D)
- **f** = ln(D + ε) — log naturel
- **f_scaled** = (f − μ_f) / σ_f avec μ_f = {stats['mu_f']:.6f}, σ_f = {stats['sigma_f']:.6f}

## Homogénéité (indicatif)

| Grandeur | Asymétrie (skewness) |
|----------|----------------------|
| D (brut) | {stats['skew_D']:.4f} |
| f        | {stats['skew_f']:.4f} |

Une skewness plus proche de 0 pour **f** que pour **D** indique une distribution plus symétrique après transformation log.

## Plages

- **D** : {stats['D_min']:.6e} — {stats['D_max']:.6e}
- **f** : {stats['f_min']:.6f} — {stats['f_max']:.6f}

## Fichiers

- Jeu pour l’entraînement (entrées + cible) : `{paths['csv']}`
- Figure histogrammes : `{paths['png']}`
- Paramètres JSON (ε, μ_f, σ_f pour inverse à l’inférence) : `{paths['json']}`

## Colonnes livrées pour le MLP

Entrée recommandée (étape 1) : **R_norm**, **cos_theta**.  
Cible : **f** (ou **f_scaled** si normalisation conservée jusqu’au modèle).

---
*Généré automatiquement par `code/scripts/etape2_transformation_cible.py`.*
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Étape 2 : f = ln(D + ε)")
    p.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "kernel_propre.csv",
        help="CSV issu de l'étape 1",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Dossier de sortie",
    )
    args = p.parse_args()

    if not args.input.is_file():
        print(f"Erreur : fichier introuvable {args.input}", file=sys.stderr)
        return 1

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_kernel_propre(args.input)
    enriched, stats = transform_target(df)

    out_csv = out_dir / "kernel_cible_log.csv"
    enriched.to_csv(out_csv, index=False)

    meta = {
        "epsilon": stats["epsilon"],
        "mu_f": stats["mu_f"],
        "sigma_f": stats["sigma_f"],
        "log": "natural_ln",
        "epsilon_rule": f"{EPSILON_FACTOR:g} * min(D)",
    }
    meta_path = out_dir / "etape2_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    png_path = out_dir / "etape2_histogrammes.png"
    try:
        _plot_histograms(df, enriched["f"].to_numpy(), png_path)
    except ImportError:
        png_path = None
        print(
            "Avertissement : matplotlib absent — pas de figure. "
            "Installez matplotlib ou ajoutez-le à requirements.txt.",
            file=sys.stderr,
        )

    def _rel(p: Path | None) -> str:
        if p is None:
            return "(non généré)"
        try:
            return str(p.relative_to(out_dir.resolve()))
        except ValueError:
            return str(p)
    paths_map = {
        "csv": _rel(out_csv),
        "png": _rel(png_path) if png_path and png_path.exists() else "(non généré)",
        "json": _rel(meta_path),
    }
    report_path = out_dir / "etape2_rapport.md"
    write_report(report_path, stats, paths_map)

    print(f"Jeu avec f, f_scaled : {out_csv} ({len(enriched)} lignes)")
    print(f"Métadonnées       : {meta_path}")
    if png_path and png_path.exists():
        print(f"Histogrammes      : {png_path}")
    print(f"Rapport           : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
