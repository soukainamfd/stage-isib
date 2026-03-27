#!/usr/bin/env python3
"""
Étape 3 du plan (code/guide/plan.md) : séparation apprentissage / validation.

- Split aléatoire 80 % / 20 % avec graine fixée (reproductibilité).
- Vérification de couverture spatiale : zones en R (nm) et bandes en θ.
- Option --stratify-spatial : stratification grossière sur les zones R pour garantir
  de la présence train/val partout (utile si le tirage pur hasard laisse un trou).

Entrée : kernel_cible_log.csv (sortie étape 2)
Sorties :
  - kernel_train.csv, kernel_val.csv
  - etape3_meta.json
  - etape3_rapport.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Zones R (nm) — alignées sur l’étape 5 du plan (profils par distance)
R_ZONE_EDGES_NM = (-np.inf, 200.0, 1000.0, np.inf)
R_ZONE_LABELS = ("R_lt_200nm", "R_200_1000nm", "R_ge_1000nm")

DEFAULT_TRAIN_FRAC = 0.8
DEFAULT_SEED = 42


def _read_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("R_nm", "theta_rad", "R_norm", "cos_theta", "f"):
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans {path}: {col}")
    return df


def _r_zone_labels(r_nm: pd.Series) -> pd.Series:
    return pd.cut(
        r_nm,
        bins=list(R_ZONE_EDGES_NM),
        labels=list(R_ZONE_LABELS),
        ordered=False,
    )


def _theta_band_labels(theta_rad: pd.Series) -> pd.Series:
    """Deux hémisphères polaires : avant / arrière du faisceau (indicatif)."""
    return pd.cut(
        theta_rad,
        bins=[-0.001, np.pi / 2, np.pi + 0.001],
        labels=["theta_0_pi2", "theta_pi2_pi"],
        ordered=False,
    )


def split_random(df: pd.DataFrame, train_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(train_frac * n))
    if n >= 2:
        n_train = max(1, min(n - 1, n_train))
    else:
        n_train = 1 if n == 1 else 0
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def split_stratified_r_zones(df: pd.DataFrame, train_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pour chaque zone R, applique un tirage 80/20 indépendant (même graine dérivée)."""
    zones = _r_zone_labels(df["R_nm"])
    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    for z in R_ZONE_LABELS:
        sub = df.loc[zones == z].copy()
        if len(sub) == 0:
            continue
        idx = rng.permutation(len(sub))
        shuffled = sub.iloc[idx].reset_index(drop=True)
        n = len(shuffled)
        n_train = int(round(train_frac * n))
        if n >= 2:
            n_train = max(1, min(n - 1, n_train))
        elif n == 1:
            n_train = 1
        train_parts.append(shuffled.iloc[:n_train])
        val_parts.append(shuffled.iloc[n_train:])
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0]
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else df.iloc[0:0]
    return train_df, val_df


def coverage_table(df: pd.DataFrame, subset_name: str) -> dict:
    rz = _r_zone_labels(df["R_nm"])
    tb = rz.value_counts(dropna=False).reindex(list(R_ZONE_LABELS), fill_value=0)
    tz = _theta_band_labels(df["theta_rad"])
    tb_theta = tz.value_counts(dropna=False)
    return {
        "subset": subset_name,
        "n": int(len(df)),
        "by_r_zone": {str(k): int(v) for k, v in tb.items()},
        "by_theta_band": {str(k): int(v) for k, v in tb_theta.items()},
        "R_nm_min": float(df["R_nm"].min()) if len(df) else None,
        "R_nm_max": float(df["R_nm"].max()) if len(df) else None,
        "theta_deg_min": float(np.rad2deg(df["theta_rad"].min())) if len(df) else None,
        "theta_deg_max": float(np.rad2deg(df["theta_rad"].max())) if len(df) else None,
    }


def check_coverage_ok(full_df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame) -> list[str]:
    """Chaque zone R présente dans le jeu complet doit avoir au moins un point en train et en val."""
    warnings: list[str] = []
    full_z = _r_zone_labels(full_df["R_nm"]).value_counts()
    tr_z = _r_zone_labels(train["R_nm"]).value_counts()
    va_z = _r_zone_labels(val["R_nm"]).value_counts()
    for z in R_ZONE_LABELS:
        if int(full_z.get(z, 0)) == 0:
            continue
        nt, nv = int(tr_z.get(z, 0)), int(va_z.get(z, 0))
        if nt == 0:
            warnings.append(
                f"Zone {z} : aucun point en train alors que le jeu complet en contient — "
                f"ajuster --train-fraction, utiliser --stratify-spatial ou changer la graine."
            )
        if nv == 0:
            warnings.append(
                f"Zone {z} : aucun point en validation alors que le jeu complet en contient — "
                f"envisager --stratify-spatial ou une graine différente."
            )
    return warnings


def write_report(
    path: Path,
    train_cov: dict,
    val_cov: dict,
    warnings: list[str],
    meta: dict,
) -> None:
    wr_lines = "\n".join(f"- {w}" for w in warnings) if warnings else "- Aucun problème signalé sur les zones R."
    text = f"""# Rapport — Étape 3 (séparation train / validation)

## Paramètres

- **Fraction train** : {meta['train_fraction']}
- **Graine** : {meta['seed']}
- **Mode split** : **{meta['split_mode']}** ({'stratifié par zones R' if meta['stratify_spatial'] else 'aléatoire global'})

## Effectifs

| Jeu | N |
|-----|---|
| Train | {train_cov['n']} |
| Validation | {val_cov['n']} |

## Couverture spatiale — distance R (zones)

Même découpe que pour l’évaluation par distance (plan, étape 5) : **R < 200 nm**, **200 nm ≤ R < 1 μm**, **R ≥ 1 μm**.

### Train

| Zone | Effectif |
|------|----------|
"""
    for k, v in train_cov["by_r_zone"].items():
        text += f"| {k} | {v} |\n"
    text += """
### Validation

| Zone | Effectif |
|------|----------|
"""
    for k, v in val_cov["by_r_zone"].items():
        text += f"| {k} | {v} |\n"
    text += f"""
## Couverture — bandes en θ (0–π/2 vs π/2–π)

### Train

| Bande | Effectif |
|-------|----------|
"""
    for k, v in train_cov["by_theta_band"].items():
        text += f"| {k} | {v} |\n"
    text += """
### Validation

| Bande | Effectif |
|-------|----------|
"""
    for k, v in val_cov["by_theta_band"].items():
        text += f"| {k} | {v} |\n"
    text += f"""
## Alertes couverture

{wr_lines}

## Voxelisation et pondération (rappel plan)

La voxelisation variable **surreprésente** les petits **R** (nombreux petits voxels). Pour l’entraînement ultérieur, on pourra **sous-échantillonner** les régions denses ou **pondérer la perte** par un volume effectif ∝ **R² ΔR Δθ** si ces pas sont disponibles ; ici les fichiers CSV sont des points sans ΔR/Δθ explicites — une approximation courante est un poids ∝ **R²** pour rééquilibrer grossièrement.

## Fichiers

- **Train** : `{meta['paths']['train_csv']}`
- **Validation** : `{meta['paths']['val_csv']}`
- **Métadonnées JSON** : `{meta['paths']['meta_json']}`

---
*Généré automatiquement par `code/scripts/etape3_train_val_split.py`.*
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Étape 3 : split train / validation + couverture spatiale")
    p.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "kernel_cible_log.csv",
        help="CSV issu de l'étape 2",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Dossier de sortie",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Graine du générateur aléatoire")
    p.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRAC,
        help="Proportion train (défaut 0.8)",
    )
    p.add_argument(
        "--stratify-spatial",
        action="store_true",
        help="Stratifier le split par zones R (<200, 200–1000 nm, ≥1000 nm)",
    )
    args = p.parse_args()

    if not (0.0 < args.train_fraction < 1.0):
        print("Erreur : --train-fraction doit être strictement entre 0 et 1.", file=sys.stderr)
        return 1

    if not args.input.is_file():
        print(f"Erreur : fichier introuvable {args.input}", file=sys.stderr)
        return 1

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_input(args.input)
    if args.stratify_spatial:
        train_df, val_df = split_stratified_r_zones(df, args.train_fraction, args.seed)
        split_mode = "stratified_r_zones"
    else:
        train_df, val_df = split_random(df, args.train_fraction, args.seed)
        split_mode = "random_shuffle"

    train_path = out_dir / "kernel_train.csv"
    val_path = out_dir / "kernel_val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    train_cov = coverage_table(train_df, "train")
    val_cov = coverage_table(val_df, "val")
    warnings = check_coverage_ok(df, train_df, val_df)

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(out_dir.resolve()))
        except ValueError:
            return str(p)

    meta = {
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "stratify_spatial": bool(args.stratify_spatial),
        "split_mode": split_mode,
        "n_total": len(df),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "r_zone_edges_nm": [None if x == -np.inf or x == np.inf else float(x) for x in R_ZONE_EDGES_NM],
        "r_zone_labels": list(R_ZONE_LABELS),
        "train_coverage": train_cov,
        "val_coverage": val_cov,
        "coverage_warnings": warnings,
        "paths": {
            "train_csv": _rel(train_path),
            "val_csv": _rel(val_path),
        },
    }
    meta_path = out_dir / "etape3_meta.json"
    meta["paths"]["meta_json"] = _rel(meta_path)
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    report_path = out_dir / "etape3_rapport.md"
    write_report(report_path, train_cov, val_cov, warnings, meta)

    print(f"Train : {train_path} ({len(train_df)} lignes)")
    print(f"Val   : {val_path} ({len(val_df)} lignes)")
    print(f"Méta  : {meta_path}")
    print(f"Rapport : {report_path}")
    for w in warnings:
        print(f"Avertissement : {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
