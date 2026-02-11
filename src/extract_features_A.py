#!/usr/bin/env python3
"""
Feature Engineering OPTIMISÉ avec Open3D natif.

Calcule automatiquement :
- linearity, planarity, sphericity (via covariances)
- verticality (via normales)

Usage:
    # Un seul fichier
    python scripts/01_preprocessing/extract_features.py \
        --input data/02_samples/micro/scene_1_micro.parquet
    
    # Tout le dossier micro (10 scènes)
    python scripts/01_preprocessing/extract_features.py \
        --input-dir data/02_samples/micro/
    
    # Tout le dossier dev
    python scripts/01_preprocessing/extract_features.py \
        --input-dir data/02_samples/dev/
"""

import argparse
import time
import numpy as np
import pandas as pd
import open3d as o3d
from pathlib import Path


def compute_geometric_features_o3d(xyz: np.ndarray, k: int = 20) -> dict:
    """
    Calcul features géométriques avec Open3D (RAPIDE).
    
    Returns:
        Dict avec linearity, planarity, sphericity, verticality
    """
    print(f"   🔍 Computing geometric features (Open3D, k={k})...")
    
    # 1. Créer point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # 2. Normales (contient aussi covariances internes)
    start = time.time()
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )
    
    # 3. Covariances (eigenvalues)
    pcd.estimate_covariances(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )
    
    elapsed = time.time() - start
    print(f"      ⏱️  Temps : {elapsed:.2f}s")
    
    # 4. Extraire normales
    normals = np.asarray(pcd.normals)
    verticality = np.abs(normals[:, 2])  # |nz| proche 0 = vertical
    
    # 5. Extraire eigenvalues depuis covariances
    covariances = np.asarray(pcd.covariances)  # Shape (N, 3, 3)
    
    linearity = np.zeros(len(xyz))
    planarity = np.zeros(len(xyz))
    sphericity = np.zeros(len(xyz))
    
    for i, cov in enumerate(covariances):
        eigvals = np.linalg.eigvalsh(cov)  # Sorted ascending
        l3, l2, l1 = eigvals[0], eigvals[1], eigvals[2]
        
        if l1 > 1e-10:
            linearity[i] = (l1 - l2) / l1
            planarity[i] = (l2 - l3) / l1
            sphericity[i] = l3 / l1
    
    return {
        'linearity': linearity.astype(np.float32),
        'planarity': planarity.astype(np.float32),
        'sphericity': sphericity.astype(np.float32),
        'verticality': verticality.astype(np.float32),
        'normal_x': normals[:, 0].astype(np.float32),
        'normal_y': normals[:, 1].astype(np.float32),
        'normal_z': normals[:, 2].astype(np.float32),
    }


def process_single_file(input_path: Path, output_dir: Path, k_neighbors: int) -> dict:
    """
    Traite un fichier parquet et retourne les stats.
    """
    output_path = output_dir / f"{input_path.stem}_features.parquet"
    
    # Skip si déjà traité
    if output_path.exists():
        print(f"   ⏭️  Déjà traité: {input_path.stem}")
        return {'status': 'skipped', 'file': input_path.stem}
    
    print(f"\n{'─'*60}")
    print(f"📊 {input_path.stem}")
    print(f"{'─'*60}")
    
    # Chargement
    df = pd.read_parquet(input_path)
    xyz = df[['x', 'y', 'z']].values
    
    print(f"   Points : {len(xyz):,}")
    
    # Calcul features
    start = time.time()
    features = compute_geometric_features_o3d(xyz, k=k_neighbors)
    elapsed = time.time() - start
    
    # Ajout colonnes
    for name, values in features.items():
        df[name] = values
    
    # Sauvegarde
    df.to_parquet(output_path, index=False)
    
    print(f"   ✅ Output : {output_path.name}")
    print(f"   ⏱️  Temps  : {elapsed:.1f}s")
    
    # Stats par classe
    stats = {}
    if 'class_id' in df.columns:
        class_data = df[df['class_id'] >= 0]
        if len(class_data) > 0:
            stats = class_data.groupby('class_id')[
                ['linearity', 'planarity', 'sphericity', 'verticality']
            ].mean().to_dict()
    
    return {
        'status': 'processed',
        'file': input_path.stem,
        'n_points': len(xyz),
        'time': elapsed,
        'stats': stats
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extraction features géométriques pour nuages de points LiDAR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Un seul fichier
  python extract_features.py --input data/02_samples/micro/scene_1_micro.parquet
  
  # Tout le dossier micro
  python extract_features.py --input-dir data/02_samples/micro/
  
  # Forcer recalcul (écrase les existants)
  python extract_features.py --input-dir data/02_samples/micro/ --force
        """
    )
    
    # Arguments mutuellement exclusifs : fichier OU dossier
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", type=Path,
                       help="Fichier parquet unique à traiter")
    group.add_argument("--input-dir", "-d", type=Path,
                       help="Dossier contenant les fichiers parquet")
    
    parser.add_argument("--output-dir", "-o", type=Path,
                        default=Path("data/03_intermediate/features"),
                        help="Dossier de sortie (défaut: data/03_intermediate/features)")
    parser.add_argument("--k-neighbors", "-k", type=int, default=20,
                        help="Nombre de voisins pour KNN (défaut: 20)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Forcer le recalcul même si fichier existe")
    
    args = parser.parse_args()
    
    # Créer dossier output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🚀 FEATURE ENGINEERING (Open3D)")
    print("=" * 70)
    print(f"   K-NN        : {args.k_neighbors}")
    print(f"   Output dir  : {args.output_dir}")
    
    # Déterminer les fichiers à traiter
    if args.input:
        # Mode fichier unique
        files = [args.input]
        print(f"   Mode        : Fichier unique")
    else:
        # Mode dossier
        files = sorted(args.input_dir.glob("*.parquet"))
        # Exclure metadata.json s'il existe en .parquet (au cas où)
        files = [f for f in files if 'metadata' not in f.stem]
        print(f"   Mode        : Dossier ({len(files)} fichiers)")
        print(f"   Input dir   : {args.input_dir}")
    
    if not files:
        print(f"\n❌ Aucun fichier .parquet trouvé")
        return 1
    
    # Si force, supprimer les fichiers existants
    if args.force:
        print(f"\n⚠️  Mode --force : écrasement des fichiers existants")
        for f in files:
            output_path = args.output_dir / f"{f.stem}_features.parquet"
            if output_path.exists():
                output_path.unlink()
    
    # Traiter chaque fichier
    results = []
    total_start = time.time()
    
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] ", end="")
        result = process_single_file(f, args.output_dir, args.k_neighbors)
        results.append(result)
    
    total_elapsed = time.time() - total_start
    
    # ══════════════════════════════════════════════════════════════
    # SYNTHÈSE
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("📊 SYNTHÈSE")
    print("=" * 70)
    
    n_processed = sum(1 for r in results if r['status'] == 'processed')
    n_skipped = sum(1 for r in results if r['status'] == 'skipped')
    
    print(f"\n   Fichiers traités : {n_processed}")
    print(f"   Fichiers ignorés : {n_skipped} (déjà existants)")
    print(f"   Temps total      : {total_elapsed:.1f}s")
    
    if n_processed > 0:
        avg_time = sum(r.get('time', 0) for r in results if r['status'] == 'processed') / n_processed
        total_points = sum(r.get('n_points', 0) for r in results if r['status'] == 'processed')
        print(f"   Temps moyen      : {avg_time:.1f}s/fichier")
        print(f"   Points totaux    : {total_points:,}")
    
    # Stats globales par classe
    print("\n📊 Features moyennes par classe (tous fichiers) :")
    
    class_names = {0: 'Antenna', 1: 'Cable', 2: 'Electric_pole', 3: 'Wind_turbine'}
    class_stats = {cid: {'linearity': [], 'planarity': [], 'sphericity': [], 'verticality': []}
                   for cid in class_names}
    
    for r in results:
        if r['status'] == 'processed' and 'stats' in r and r['stats']:
            for feat in ['linearity', 'planarity', 'sphericity', 'verticality']:
                if feat in r['stats']:
                    for cid, val in r['stats'][feat].items():
                        if cid in class_stats:
                            class_stats[cid][feat].append(val)
    
    print(f"\n   {'Classe':<15} | {'Linearity':>10} | {'Planarity':>10} | {'Sphericity':>10} | {'Verticality':>10}")
    print("   " + "-" * 65)
    
    for cid, name in class_names.items():
        if class_stats[cid]['linearity']:
            lin = np.mean(class_stats[cid]['linearity'])
            pla = np.mean(class_stats[cid]['planarity'])
            sph = np.mean(class_stats[cid]['sphericity'])
            ver = np.mean(class_stats[cid]['verticality'])
            print(f"   {name:<15} | {lin:>10.3f} | {pla:>10.3f} | {sph:>10.3f} | {ver:>10.3f}")
    
    # Vérification finale
    print(f"\n📂 Fichiers générés dans {args.output_dir}/ :")
    output_files = sorted(args.output_dir.glob("*_features.parquet"))
    for f in output_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"   ✓ {f.name} ({size_mb:.1f} MB)")
    
    print("\n" + "=" * 70)
    print("✅ TERMINÉ")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
