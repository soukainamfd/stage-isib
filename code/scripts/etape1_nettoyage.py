#!/usr/bin/env python3
"""
Étape 1 du plan (code/guide/plan.md) : nettoyage et préparation des données kernel D(R,θ).

Entrée CSV attendue (séparateur , ou ;) :
  voxel_id, R_nm, theta_rad, D, sigma
  (theta_rad = angle polaire depuis l'axe z du faisceau ; R_nm distance au centre NP)

Sorties :
  - kernel_propre.csv : jeu filtré + colonnes R_norm, cos_theta, x_norm, z_norm
  - etape1_rapport.md : statistiques et choix de coordonnées
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R_NP_NM = 50.0  # rayon NP, diamètre 100 nm (thèse Derrien)
SIGMA_REL_MAX = 0.05


def _read_table(path: Path) -> pd.DataFrame:
    for sep in (",", ";", "\t"):
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) >= 2:
                return df
        except Exception:
            continue
    return pd.read_csv(path)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mappe des noms de colonnes usuels vers le schéma canonique."""
    lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for key, target in [
        ("voxel", "voxel_id"),
        ("voxel_id", "voxel_id"),
        ("id", "voxel_id"),
        ("r", "R_nm"),
        ("r_nm", "R_nm"),
        ("theta", "theta_rad"),
        ("theta_rad", "theta_rad"),
        ("theta_deg", "theta_deg"),
        ("dose", "D"),
        ("d", "D"),
        ("sigma", "sigma"),
        ("err", "sigma"),
        ("uncertainty", "sigma"),
    ]:
        if key in lower and target not in rename.values():
            rename[lower[key]] = target
    out = df.rename(columns=rename)
    if "theta_deg" in out.columns and "theta_rad" not in out.columns:
        out["theta_rad"] = np.deg2rad(out["theta_deg"])
    required = {"R_nm", "theta_rad", "D"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes après normalisation : {missing}. "
            f"Colonnes trouvées : {list(out.columns)}"
        )
    if "sigma" not in out.columns:
        out["sigma"] = np.nan
    if "voxel_id" not in out.columns:
        out["voxel_id"] = np.arange(1, len(out) + 1)
    return out


def create_synthetic_example(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    """Jeu factice pour démo (en attendant les sorties Monte Carlo réelles)."""
    rng = np.random.default_rng(seed)
    R = np.exp(rng.uniform(np.log(50.0), np.log(7000.0), n))
    theta = rng.uniform(0.0, np.pi, n)
    base = 1e6 / (R**2 + 100.0) * (1.0 + 0.35 * np.cos(theta))
    noise = rng.uniform(0.92, 1.08, n)
    D = np.maximum(base * noise, 1e-12)
    rel_err = rng.uniform(0.005, 0.08, n)
    sigma = rel_err * D
    return pd.DataFrame(
        {
            "voxel_id": np.arange(1, n + 1),
            "R_nm": R,
            "theta_rad": theta,
            "D": D,
            "sigma": sigma,
        }
    )


def clean_and_augment(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Filtres étape 1 + features Option A / B."""
    n0 = len(df)
    stats: dict = {"n_initial": n0}

    df = df.copy()
    df = df[df["D"] > 0].copy()
    stats["apres_D_positif"] = len(df)

    mask_sigma = df["sigma"].notna() & (df["D"] > 0)
    rel = pd.Series(np.nan, index=df.index)
    rel.loc[mask_sigma] = df.loc[mask_sigma, "sigma"] / df.loc[mask_sigma, "D"]
    df["_rel_err"] = rel
    df = df[(df["_rel_err"].isna()) | (df["_rel_err"] <= SIGMA_REL_MAX)].copy()
    stats["apres_sigma_rel_5pct"] = len(df)

    # Doublons (R, θ) — arrondi nm + 1e-4 rad
    key_r = df["R_nm"].round(3)
    key_t = df["theta_rad"].round(4)
    dup = df.assign(_kr=key_r, _kt=key_t).duplicated(subset=["_kr", "_kt"], keep="first")
    stats["doublons_supprimes"] = int(dup.sum())
    df = df.loc[~dup].drop(columns=["_kr", "_kt"], errors="ignore")

    df["R_norm"] = df["R_nm"] / R_NP_NM
    df["cos_theta"] = np.cos(df["theta_rad"])
    df["x_norm"] = (df["R_nm"] * np.sin(df["theta_rad"])) / R_NP_NM
    df["z_norm"] = (df["R_nm"] * np.cos(df["theta_rad"])) / R_NP_NM

    df = df.drop(columns=["_rel_err"], errors="ignore")

    stats["n_final"] = len(df)
    stats["R_nm_min"] = float(df["R_nm"].min())
    stats["R_nm_max"] = float(df["R_nm"].max())
    stats["theta_deg_min"] = float(np.rad2deg(df["theta_rad"].min()))
    stats["theta_deg_max"] = float(np.rad2deg(df["theta_rad"].max()))
    stats["D_min"] = float(df["D"].min())
    stats["D_max"] = float(df["D"].max())
    stats["log10_D_range"] = float(np.log10(stats["D_max"] / max(stats["D_min"], 1e-300)))

    return df, stats


def write_report(path: Path, stats: dict) -> None:
    text = f"""# Rapport — Étape 1 (nettoyage et préparation)

## Filtres appliqués

- Conservation des voxels avec **D > 0**.
- Exclusion des lignes avec **σ/D > {SIGMA_REL_MAX:.0%}** (lignes sans σ conservées).
- Suppression des **doublons** (même R arrondi à 0,001 nm et θ arrondi à 1e-4 rad).

## Effectifs

| Étape | Nombre de lignes |
|------|------------------|
| Initial | {stats['n_initial']} |
| Après D > 0 | {stats.get('apres_D_positif', '—')} |
| Après filtre σ/D | {stats.get('apres_sigma_rel_5pct', '—')} |
| Doublons supprimés | {stats.get('doublons_supprimes', 0)} |
| **Final** | **{stats['n_final']}** |

## Plages observées (jeu propre)

- **R** : {stats['R_nm_min']:.3f} — {stats['R_nm_max']:.3f} nm  
- **θ** : {stats['theta_deg_min']:.2f}° — {stats['theta_deg_max']:.2f}°  
- **D** : {stats['D_min']:.6e} — {stats['D_max']:.6e} (unités du fichier source)  
- **Étendue log₁₀(D)** : ~{stats['log10_D_range']:.2f} ordres de grandeur  

## Choix du système de coordonnées pour le MLP (recommandation plan)

**Option A retenue** : entrée **(R_norm, cos θ)** avec `R_norm = R / R_NP` et `R_NP = {R_NP_NM:g} nm`.

- Cohérent avec la symétrie azimutale du problème (faisceau selon **z**).
- `cos θ` évite la singularité de représentation aux pôles comparé à θ seul.

Les colonnes **x_norm**, **z_norm** (Option B) sont ajoutées pour visualisation et tests ultérieurs.

---
*Généré automatiquement par `code/scripts/etape1_nettoyage.py`.*
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Étape 1 : nettoyage kernel D(R,θ)")
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="CSV Monte Carlo (colonnes voxel_id, R_nm, theta_rad, D, sigma). "
        "Si absent : génère un jeu synthétique de démonstration.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Dossier de sortie",
    )
    p.add_argument("--synthetic-n", type=int, default=8000, help="Taille du jeu synthétique si --input absent")
    args = p.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input is None or not args.input.is_file():
        if args.input is not None:
            print(f"Avertissement : fichier introuvable {args.input}, jeu synthétique utilisé.", file=sys.stderr)
        raw = create_synthetic_example(n=args.synthetic_n)
        raw_path = out_dir / "kernel_brut_exemple.csv"
        raw.to_csv(raw_path, index=False)
        print(f"Jeu synthétique écrit : {raw_path}")
    else:
        raw = _normalize_columns(_read_table(args.input))

    propre, stats = clean_and_augment(raw)

    out_csv = out_dir / "kernel_propre.csv"
    propre.to_csv(out_csv, index=False)
    report_path = out_dir / "etape1_rapport.md"
    write_report(report_path, stats)

    print(f"Jeu propre : {out_csv} ({stats['n_final']} lignes)")
    print(f"Rapport    : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
