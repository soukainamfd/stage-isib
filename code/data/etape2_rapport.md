# Rapport — Étape 2 (transformation de la cible)

## Paramètres

- **ε** = 1e-06 × min(D) = **1.328199e-08** (mêmes unités que D)
- **f** = ln(D + ε) — log naturel
- **f_scaled** = (f − μ_f) / σ_f avec μ_f = 1.063122, σ_f = 2.870955

## Homogénéité (indicatif)

| Grandeur | Asymétrie (skewness) |
|----------|----------------------|
| D (brut) | 2.7700 |
| f        | -0.0235 |

Une skewness plus proche de 0 pour **f** que pour **D** indique une distribution plus symétrique après transformation log.

## Plages

- **D** : 1.328199e-02 — 5.431442e+02
- **f** : -4.321346 — 6.297375

## Fichiers

- Jeu pour l’entraînement (entrées + cible) : `kernel_cible_log.csv`
- Figure histogrammes : `etape2_histogrammes.png`
- Paramètres JSON (ε, μ_f, σ_f pour inverse à l’inférence) : `etape2_meta.json`

## Colonnes livrées pour le MLP

Entrée recommandée (étape 1) : **R_norm**, **cos_theta**.  
Cible : **f** (ou **f_scaled** si normalisation conservée jusqu’au modèle).

---
*Généré automatiquement par `code/scripts/etape2_transformation_cible.py`.*
