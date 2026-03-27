# Rapport — Étape 3 (séparation train / validation)

## Paramètres

- **Fraction train** : 0.8
- **Graine** : 42
- **Mode split** : **random_shuffle** (aléatoire global)

## Effectifs

| Jeu | N |
|-----|---|
| Train | 3810 |
| Validation | 952 |

## Couverture spatiale — distance R (zones)

Même découpe que pour l’évaluation par distance (plan, étape 5) : **R < 200 nm**, **200 nm ≤ R < 1 μm**, **R ≥ 1 μm**.

### Train

| Zone | Effectif |
|------|----------|
| R_lt_200nm | 1128 |
| R_200_1000nm | 1184 |
| R_ge_1000nm | 1498 |

### Validation

| Zone | Effectif |
|------|----------|
| R_lt_200nm | 266 |
| R_200_1000nm | 322 |
| R_ge_1000nm | 364 |

## Couverture — bandes en θ (0–π/2 vs π/2–π)

### Train

| Bande | Effectif |
|-------|----------|
| theta_pi2_pi | 1937 |
| theta_0_pi2 | 1873 |

### Validation

| Bande | Effectif |
|-------|----------|
| theta_0_pi2 | 479 |
| theta_pi2_pi | 473 |

## Alertes couverture

- Aucun problème signalé sur les zones R.

## Voxelisation et pondération (rappel plan)

La voxelisation variable **surreprésente** les petits **R** (nombreux petits voxels). Pour l’entraînement ultérieur, on pourra **sous-échantillonner** les régions denses ou **pondérer la perte** par un volume effectif ∝ **R² ΔR Δθ** si ces pas sont disponibles ; ici les fichiers CSV sont des points sans ΔR/Δθ explicites — une approximation courante est un poids ∝ **R²** pour rééquilibrer grossièrement.

## Fichiers

- **Train** : `kernel_train.csv`
- **Validation** : `kernel_val.csv`
- **Métadonnées JSON** : `etape3_meta.json`

---
*Généré automatiquement par `code/scripts/etape3_train_val_split.py`.*
