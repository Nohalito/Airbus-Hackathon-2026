#!/usr/bin/env python3
"""
Valide les paramètres RANSAC/DBSCAN sur les échantillons.
Génère un rapport détaillé + recommandations.

Usage:
    # Tous les fichiers (défaut)
    python scripts/01_preprocessing/validate_all_scenes.py
    
    # Seulement N fichiers (test rapide)
    python scripts/01_preprocessing/validate_all_scenes.py --limit 2
    
    # Fichiers spécifiques
    python scripts/01_preprocessing/validate_all_scenes.py --files scene_1 scene_3
    
    # Pattern glob
    python scripts/01_preprocessing/validate_all_scenes.py --pattern "scene_1*"
    
    # Autre dossier
    python scripts/01_preprocessing/validate_all_scenes.py --samples data/02_samples/dev/
"""

import argparse
import pandas as pd
import open3d as o3d
import numpy as np
from pathlib import Path
import yaml
import gc
from dataclasses import dataclass
from typing import List
from collections import Counter

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

CONFIG_PATH = Path("config/optimal_params.yaml")
DEFAULT_SAMPLES_DIR = Path("data/02_samples/micro")

# Valeurs par défaut
DEFAULT_RANSAC = 0.2
DEFAULT_EPS = 1.5
DEFAULT_MIN_POINTS = 10

# Seuils de validation
VALID_SOL_MIN = 20    # % minimum sol détecté
VALID_SOL_MAX = 80    # % maximum sol détecté
VALID_CLUSTERS_MIN = 1
VALID_CLUSTERS_MAX = 200
VALID_NOISE_MAX = 50  # % maximum bruit


@dataclass
class SceneResult:
    name: str
    n_points: int
    sol_pct: float
    n_obstacles: int
    n_clusters: int
    noise_pct: float
    status: str
    warnings: List[str]


# ══════════════════════════════════════════════════════════════
# CHARGEMENT CONFIG
# ══════════════════════════════════════════════════════════════

def load_config():
    """Charge les paramètres depuis le YAML ou utilise les défauts."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extraction (gère plusieurs formats possibles)
        try:
            ransac = config['preprocessing']['ransac']['distance_threshold']
        except (KeyError, TypeError):
            try:
                ransac = config['preprocessing']['ransac_ground_removal']['distance_threshold']
            except (KeyError, TypeError):
                ransac = DEFAULT_RANSAC
        
        try:
            eps = config['clustering']['dbscan']['eps']
            min_pts = config['clustering']['dbscan']['min_points']
        except (KeyError, TypeError):
            try:
                eps = config['clustering']['eps']
                min_pts = config['clustering']['min_points']
            except (KeyError, TypeError):
                eps = DEFAULT_EPS
                min_pts = DEFAULT_MIN_POINTS
        
        return ransac, eps, min_pts
    else:
        print(f"⚠️ Config non trouvée ({CONFIG_PATH}), valeurs par défaut")
        return DEFAULT_RANSAC, DEFAULT_EPS, DEFAULT_MIN_POINTS


# ══════════════════════════════════════════════════════════════
# RÉSOLUTION DES FICHIERS
# ══════════════════════════════════════════════════════════════

def resolve_files(samples_dir: Path, limit: int = None,
                  files: list = None, pattern: str = None) -> list:
    """
    Résout la liste des fichiers à traiter.
    
    Args:
        samples_dir: Dossier contenant les .parquet
        limit: Nombre max de fichiers (None = tous)
        files: Liste de noms spécifiques (sans extension)
        pattern: Pattern glob (ex: "scene_1*")
    
    Returns:
        Liste de Path des fichiers à traiter
    """
    # Cas 1 : Fichiers spécifiques
    if files:
        result = []
        for name in files:
            matches = list(samples_dir.glob(f"{name}*.parquet"))
            if matches:
                result.extend(matches)
            else:
                print(f"⚠️ Fichier non trouvé : {name}")
        return sorted(set(result))
    
    # Cas 2 : Pattern glob
    if pattern:
        if not pattern.endswith('.parquet'):
            pattern = f"{pattern}*.parquet" if not pattern.endswith('*') else f"{pattern}.parquet"
        return sorted(samples_dir.glob(pattern))
    
    # Cas 3 : Tous les fichiers (avec limite optionnelle)
    all_files = sorted(samples_dir.glob("*.parquet"))
    
    if limit:
        return all_files[:limit]
    
    return all_files


# ══════════════════════════════════════════════════════════════
# VALIDATION D'UNE SCÈNE
# ══════════════════════════════════════════════════════════════

def validate_scene(parquet_file: Path, ransac_thresh: float,
                   dbscan_eps: float, dbscan_min: int) -> SceneResult:
    """
    Valide une scène avec les paramètres.
    
    Returns:
        SceneResult avec métriques + diagnostic
    """
    print(f"🔍 {parquet_file.stem:<30}", end=" ", flush=True)
    
    warnings = []
    
    try:
        # 1. Chargement
        df = pd.read_parquet(parquet_file)
        xyz = df[['x', 'y', 'z']].values
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        
        n_points_total = len(pcd.points)
        
        # 2. RANSAC (Sol)
        _, inliers = pcd.segment_plane(
            distance_threshold=ransac_thresh,
            ransac_n=3,
            num_iterations=500
        )
        
        sol_pct = len(inliers) / n_points_total * 100
        obstacle_cloud = pcd.select_by_index(inliers, invert=True)
        n_obstacles = len(obstacle_cloud.points)
        
        # 3. DBSCAN (Obstacles)
        n_clusters = 0
        noise_pct = 0
        
        if n_obstacles > 0:
            labels = np.array(obstacle_cloud.cluster_dbscan(
                eps=dbscan_eps,
                min_points=dbscan_min,
                print_progress=False
            ))
            
            n_clusters = labels.max() + 1
            n_noise = (labels == -1).sum()
            noise_pct = (n_noise / len(labels)) * 100 if len(labels) > 0 else 0
        
        # 4. Diagnostic
        status = "✅ OK"
        
        if sol_pct < VALID_SOL_MIN:
            warnings.append(f"Peu de sol ({sol_pct:.0f}%)")
            status = "⚠️ PEU SOL"
        
        if sol_pct > VALID_SOL_MAX:
            warnings.append(f"Trop de sol ({sol_pct:.0f}%)")
            status = "⚠️ TROP SOL"
        
        if n_clusters == 0 and n_obstacles > 100:
            warnings.append("0 clusters")
            status = "❌ 0 CLUSTER"
        
        if n_clusters > VALID_CLUSTERS_MAX:
            warnings.append(f"Trop de clusters ({n_clusters})")
            status = "⚠️ FRAGMENTÉ"
        
        if noise_pct > VALID_NOISE_MAX:
            warnings.append(f"Trop de bruit ({noise_pct:.0f}%)")
            status = "⚠️ BRUIT"
        
        print(f"→ {status}")
        
        result = SceneResult(
            name=parquet_file.stem,
            n_points=n_points_total,
            sol_pct=sol_pct,
            n_obstacles=n_obstacles,
            n_clusters=n_clusters,
            noise_pct=noise_pct,
            status=status,
            warnings=warnings
        )
        
        # Nettoyage RAM
        del df, pcd, obstacle_cloud
        gc.collect()
        
        return result
    
    except Exception as e:
        print(f"→ ❌ ERREUR : {str(e)}")
        return SceneResult(
            name=parquet_file.stem,
            n_points=0,
            sol_pct=0,
            n_obstacles=0,
            n_clusters=0,
            noise_pct=0,
            status="❌ ERREUR",
            warnings=[str(e)]
        )


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Valide les paramètres RANSAC/DBSCAN sur les échantillons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Tous les fichiers
  python validate_all_scenes.py
  
  # Test rapide (2 fichiers)
  python validate_all_scenes.py --limit 2
  
  # Fichiers spécifiques
  python validate_all_scenes.py --files scene_1 scene_3
  
  # Pattern
  python validate_all_scenes.py --pattern "scene_1*"
  
  # Autre dossier
  python validate_all_scenes.py --samples data/02_samples/dev/
        """
    )
    
    parser.add_argument("--samples", "-s", type=Path, default=DEFAULT_SAMPLES_DIR,
                        help=f"Dossier des échantillons (défaut: {DEFAULT_SAMPLES_DIR})")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Nombre max de fichiers à traiter")
    parser.add_argument("--files", "-f", nargs='+', default=None,
                        help="Noms des fichiers spécifiques (sans extension)")
    parser.add_argument("--pattern", "-p", type=str, default=None,
                        help="Pattern glob (ex: 'scene_1*')")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🧪 VALIDATION MULTI-SCÈNES")
    print("=" * 80)
    
    # Vérifier dossier
    if not args.samples.exists():
        print(f"\n❌ Dossier non trouvé : {args.samples}")
        print(f"   Lancez : python scripts/00_setup/create_samples.py --type micro")
        return 1
    
    # Charger config
    RANSAC_THRESH, DBSCAN_EPS, DBSCAN_MIN = load_config()
    print(f"\n✅ Paramètres : RANSAC={RANSAC_THRESH}m, DBSCAN_EPS={DBSCAN_EPS}m, MIN_PTS={DBSCAN_MIN}")
    
    # Résoudre fichiers
    files = resolve_files(
        samples_dir=args.samples,
        limit=args.limit,
        files=args.files,
        pattern=args.pattern
    )
    
    if not files:
        print(f"\n❌ Aucun fichier trouvé dans {args.samples}")
        return 1
    
    # Afficher mode
    mode = "TOUS"
    if args.limit:
        mode = f"LIMIT {args.limit}"
    elif args.files:
        mode = f"FILES {args.files}"
    elif args.pattern:
        mode = f"PATTERN '{args.pattern}'"
    
    print(f"\n📂 {len(files)} scène(s) à valider (mode: {mode})\n")
    
    # Header tableau
    print(f"{'SCÈNE':<30} | {'POINTS':<8} | {'SOL %':<7} | {'CLUST.':<7} | {'BRUIT':<7} | {'VERDICT'}")
    print("-" * 95)
    
    # Validation
    results = []
    for f in files:
        res = validate_scene(f, RANSAC_THRESH, DBSCAN_EPS, DBSCAN_MIN)
        results.append(res)
        
        # Ligne tableau
        print(f"{res.name:<30} | {res.n_points:>8,} | {res.sol_pct:>6.1f}% | "
              f"{res.n_clusters:>7} | {res.noise_pct:>6.1f}% | {res.status}")
    
    # ══════════════════════════════════════════════════════════
    # SYNTHÈSE
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("📊 SYNTHÈSE")
    print("=" * 80)
    
    n_ok = sum(1 for r in results if r.status == "✅ OK")
    n_warn = sum(1 for r in results if "⚠️" in r.status)
    n_err = sum(1 for r in results if "❌" in r.status)
    
    print(f"\n✅ OK       : {n_ok}/{len(results)} ({n_ok/len(results)*100:.0f}%)")
    print(f"⚠️  Warnings : {n_warn}/{len(results)}")
    print(f"❌ Erreurs  : {n_err}/{len(results)}")
    
    # Statistiques globales
    valid_results = [r for r in results if r.n_points > 0]
    
    if valid_results:
        sol_pcts = [r.sol_pct for r in valid_results]
        clusters = [r.n_clusters for r in valid_results]
        noise_pcts = [r.noise_pct for r in valid_results]
        
        print(f"\n📈 MÉTRIQUES GLOBALES :")
        print(f"   Sol détecté    : {np.mean(sol_pcts):.1f}% (±{np.std(sol_pcts):.1f}%)")
        print(f"                    min={np.min(sol_pcts):.1f}%, max={np.max(sol_pcts):.1f}%")
        print(f"   Clusters/scène : {np.mean(clusters):.1f} (±{np.std(clusters):.1f})")
        print(f"                    min={np.min(clusters)}, max={np.max(clusters)}")
        print(f"   Bruit moyen    : {np.mean(noise_pcts):.1f}%")
    
    # Warnings groupés
    all_warnings = [w for r in results for w in r.warnings]
    if all_warnings:
        print(f"\n⚠️  PROBLÈMES DÉTECTÉS ({len(all_warnings)}) :")
        warning_counts = Counter(all_warnings)
        for warn, count in warning_counts.most_common(5):
            print(f"   • {warn} ({count}x)")
    
    # ══════════════════════════════════════════════════════════
    # RECOMMANDATIONS
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("💡 RECOMMANDATIONS")
    print("=" * 80 + "\n")
    
    if n_ok == len(results):
        print("🎉 Paramètres validés sur TOUTES les scènes !")
        print("   → Prêt pour l'étape suivante (extraction features)")
    else:
        if valid_results:
            mean_sol = np.mean(sol_pcts)
            mean_noise = np.mean(noise_pcts)
            mean_clusters = np.mean(clusters)
            
            if mean_sol < VALID_SOL_MIN:
                print(f"⚠️  Sol sous-détecté (moyenne {mean_sol:.0f}%)")
                print(f"   → Essayez RANSAC_THRESHOLD = {RANSAC_THRESH + 0.1:.1f}m")
            
            elif mean_sol > VALID_SOL_MAX:
                print(f"⚠️  Sol sur-détecté (moyenne {mean_sol:.0f}%)")
                print(f"   → Essayez RANSAC_THRESHOLD = {RANSAC_THRESH - 0.05:.2f}m")
            
            if mean_clusters > VALID_CLUSTERS_MAX:
                print(f"⚠️  Trop de fragmentation (moyenne {mean_clusters:.0f} clusters)")
                print(f"   → Essayez DBSCAN_EPS = {DBSCAN_EPS + 0.5:.1f}m")
            
            if mean_noise > VALID_NOISE_MAX:
                print(f"⚠️  Trop de bruit (moyenne {mean_noise:.0f}%)")
                print(f"   → Essayez DBSCAN_EPS = {DBSCAN_EPS + 0.3:.1f}m ou MIN_PTS = {max(5, DBSCAN_MIN - 3)}")
        
        if n_err > 0:
            print(f"❌ {n_err} scène(s) en erreur → Vérifiez les fichiers")
    
    print("\n" + "=" * 80 + "\n")
    
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    exit(main())
