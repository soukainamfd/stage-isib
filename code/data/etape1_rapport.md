# Rapport — Étape 1 (nettoyage et préparation)

## Filtres appliqués

- Conservation des voxels avec **D > 0**.
- Exclusion des lignes avec **σ/D > 5%** (lignes sans σ conservées).
- Suppression des **doublons** (même R arrondi à 0,001 nm et θ arrondi à 1e-4 rad).

## Effectifs

| Étape | Nombre de lignes |
|------|------------------|
| Initial | 8000 |
| Après D > 0 | 8000 |
| Après filtre σ/D | 4762 |
| Doublons supprimés | 0 |
| **Final** | **4762** |

## Plages observées (jeu propre)

- **R** : 50.128 — 6990.143 nm  
- **θ** : 0.02° — 179.99°  
- **D** : 1.328199e-02 — 5.431442e+02 (unités du fichier source)  
- **Étendue log₁₀(D)** : ~4.61 ordres de grandeur  

## Choix du système de coordonnées pour le MLP (recommandation plan)

**Option A retenue** : entrée **(R_norm, cos θ)** avec `R_norm = R / R_NP` et `R_NP = 50 nm`.

- Cohérent avec la symétrie azimutale du problème (faisceau selon **z**).
- `cos θ` évite la singularité de représentation aux pôles comparé à θ seul.

Les colonnes **x_norm**, **z_norm** (Option B) sont ajoutées pour visualisation et tests ultérieurs.

---
*Généré automatiquement par `code/scripts/etape1_nettoyage.py`.*
