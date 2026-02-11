#!/usr/bin/env python3
"""
Extrait les Bounding Boxes GROUND TRUTH depuis les labels RGB.

Usage:
    # Un seul fichier
    python scripts/01_preprocessing/extract_bboxes_gt.py \
        --input data/02_samples/micro/scene_1_micro.parquet
    
    # Tout le dossier micro (10 scènes)
    python scripts/01_preprocessing/extract_bboxes_gt.py \
        --input-dir data/02_samples/micro/
"""

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from sklearn.cluster import DBSCAN
import time

CLASS_NAMES = {
    0: "Antenna",
    1: "Cable",
    2: "Electric_pole",
    3: "Wind_turbine"
}

# Paramètres DBSCAN adaptés au type d'échantillon
CLUSTERING_PARAMS = {
    'micro': {  # 10% sampling - points espacés
        0: {'eps': 5.0, 'min_samples': 3},   # Antenna
        1: {'eps': 8.0, 'min_samples': 2},   # Cable
        2: {'eps': 5.0, 'min_samples': 5},   # Electric pole
        3: {'eps': 10.0, 'min_samples': 5},  # Wind turbine
        'min_points_bbox': 3
    },
    'dev': {  # 30% sampling
        0: {'eps': 3.0, 'min_samples': 5},
        1: {'eps': 5.0, 'min_samples': 5},
        2: {'eps': 3.0, 'min_samples': 8},
        3: {'eps': 5.0, 'min_samples': 8},
        'min_points_bbox': 5
    },
    'full': {  # 100% des points
        0: {'eps': 1.5, 'min_samples': 10},
        1: {'eps': 3.0, 'min_samples': 10},
        2: {'eps': 1.5, 'min_samples': 10},
        3: {'eps': 2.0, 'min_samples': 10},
        'min_points_bbox': 10
    }
}


def compute_oriented_bbox(xyz: np.ndarray) -> dict:
    """Calcule une Oriented Bounding Box via PCA."""
    if len(xyz) < 3:
        center = xyz.mean(axis=0) if len(xyz) > 0 else np.zeros(3)
        extent = xyz.max(axis=0) - xyz.min(axis=0) if len(xyz) > 1 else np.array([0.1, 0.1, 0.1])
        extent = np.maximum(extent, 0.1)
        return {
            'center': center,
            'extent': extent,
            'rotation_matrix': np.eye(3),
            'yaw': 0.0
        }
    
    center = xyz.mean(axis=0)
    centered = xyz - center
    
    # PCA
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, order]
    
    # Projeter
    projected = centered @ eigenvectors
    min_coords = projected.min(axis=0)
    max_coords = projected.max(axis=0)
    extent = max_coords - min_coords
    extent = np.maximum(extent, 0.1)
    
    # Recalculer centre
    local_center = (min_coords + max_coords) / 2
    center = center + eigenvectors @ local_center
    
    R = eigenvectors
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    
    yaw = np.arctan2(R[1, 0], R[0, 0])
    
    return {
        'center': center,
        'extent': extent,
        'rotation_matrix': R,
        'yaw': yaw
    }


def detect_sample_type(filepath: Path) -> str:
    """Détecte le type d'échantillon depuis le nom du fichier."""
    name = filepath.stem.lower()
    if 'micro' in name:
        return 'micro'
    elif 'dev' in name:
        return 'dev'
    else:
        return 'full'


def extract_gt_bboxes(df: pd.DataFrame, sample_type: str = 'micro') -> list:
    """
    Extrait bboxes GT avec paramètres adaptés au type d'échantillon.
    """
    params = CLUSTERING_PARAMS.get(sample_type, CLUSTERING_PARAMS['micro'])
    min_points_bbox = params['min_points_bbox']
    
    bboxes = []
    
    for class_id in CLASS_NAMES.keys():
        class_df = df[df['class_id'] == class_id]
        n_points = len(class_df)
        
        if n_points < min_points_bbox:
            if n_points > 0:
                print(f"      {CLASS_NAMES[class_id]:15s} : SKIP ({n_points} pts < {min_points_bbox})")
            continue
        
        xyz = class_df[['x', 'y', 'z']].values.astype(np.float64)
        
        # Paramètres DBSCAN
        eps = params[class_id]['eps']
        min_samples = params[class_id]['min_samples']
        
        # DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(xyz)
        
        n_instances = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        # Si aucun cluster mais assez de points → créer 1 bbox globale
        if n_instances == 0 and n_points >= min_points_bbox:
            print(f"      {CLASS_NAMES[class_id]:15s} : 1 bbox (fallback), {n_points} pts")
            obb = compute_oriented_bbox(xyz)
            bboxes.append({
                'class_id': int(class_id),
                'class_name': CLASS_NAMES[class_id],
                'instance_id': 0,
                'n_points': int(n_points),
                'center_x': float(obb['center'][0]),
                'center_y': float(obb['center'][1]),
                'center_z': float(obb['center'][2]),
                'width': float(obb['extent'][0]),
                'length': float(obb['extent'][1]),
                'height': float(obb['extent'][2]),
                'yaw': float(obb['yaw']),
                'rotation_matrix': obb['rotation_matrix'].tolist()
            })
        else:
            print(f"      {CLASS_NAMES[class_id]:15s} : {n_instances} instance(s), {n_points} pts, noise={n_noise}")
            
            # Créer bbox pour chaque instance
            for instance_id in range(labels.max() + 1):
                mask = labels == instance_id
                instance_xyz = xyz[mask]
                
                if len(instance_xyz) < min_points_bbox:
                    continue
                
                obb = compute_oriented_bbox(instance_xyz)
                
                bboxes.append({
                    'class_id': int(class_id),
                    'class_name': CLASS_NAMES[class_id],
                    'instance_id': int(instance_id),
                    'n_points': int(mask.sum()),
                    'center_x': float(obb['center'][0]),
                    'center_y': float(obb['center'][1]),
                    'center_z': float(obb['center'][2]),
                    'width': float(obb['extent'][0]),
                    'length': float(obb['extent'][1]),
                    'height': float(obb['extent'][2]),
                    'yaw': float(obb['yaw']),
                    'rotation_matrix': obb['rotation_matrix'].tolist()
                })
    
    return bboxes


def process_single_file(input_path: Path, output_dir: Path, sample_type: str = 'auto') -> dict:
    """Traite un fichier parquet et retourne les stats."""
    output_path = output_dir / f"{input_path.stem}_gt.json"
    
    # Skip si déjà traité
    if output_path.exists():
        # Lire le nombre de bboxes existantes
        with open(output_path) as f:
            existing = json.load(f)
        print(f"   ⏭️  Déjà traité: {input_path.stem} ({existing['n_bboxes']} bboxes)")
        return {'status': 'skipped', 'file': input_path.stem, 'n_bboxes': existing['n_bboxes']}
    
    # Détection auto du type
    if sample_type == 'auto':
        sample_type = detect_sample_type(input_path)
    
    print(f"\n{'─'*60}")
    print(f"📦 {input_path.stem}")
    print(f"{'─'*60}")
    print(f"   Type : {sample_type}")
    
    # Chargement
    df = pd.read_parquet(input_path)
    
    # Stats classes
    print(f"   Classes présentes :")
    for cid, name in CLASS_NAMES.items():
        count = (df['class_id'] == cid).sum()
        if count > 0:
            print(f"      {name:15s} : {count:,} points")
    
    # Extraction
    print(f"   Extraction :")
    start = time.time()
    bboxes = extract_gt_bboxes(df, sample_type=sample_type)
    elapsed = time.time() - start
    
    # Sauvegarde
    with open(output_path, 'w') as f:
        json.dump({
            'source_file': str(input_path),
            'sample_type': sample_type,
            'n_bboxes': len(bboxes),
            'bboxes': bboxes
        }, f, indent=2)
    
    print(f"   ✅ Output : {output_path.name} ({len(bboxes)} bboxes)")
    print(f"   ⏱️  Temps  : {elapsed:.2f}s")
    
    return {
        'status': 'processed',
        'file': input_path.stem,
        'n_bboxes': len(bboxes),
        'time': elapsed,
        'bboxes_by_class': {CLASS_NAMES[b['class_id']]: 0 for b in bboxes}
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extraction Bounding Boxes Ground Truth depuis labels RGB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Un seul fichier
  python extract_bboxes_gt.py --input data/02_samples/micro/scene_1_micro.parquet
  
  # Tout le dossier micro
  python extract_bboxes_gt.py --input-dir data/02_samples/micro/
  
  # Forcer recalcul
  python extract_bboxes_gt.py --input-dir data/02_samples/micro/ --force
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", type=Path,
                       help="Fichier parquet unique à traiter")
    group.add_argument("--input-dir", "-d", type=Path,
                       help="Dossier contenant les fichiers parquet")
    
    parser.add_argument("--output-dir", "-o", type=Path,
                        default=Path("data/03_intermediate/bboxes_gt"),
                        help="Dossier de sortie")
    parser.add_argument("--sample-type", "-t", choices=['micro', 'dev', 'full', 'auto'],
                        default='auto', help="Type d'échantillon (défaut: auto)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Forcer le recalcul même si fichier existe")
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("📦 EXTRACTION BOUNDING BOXES GROUND TRUTH")
    print("=" * 70)
    print(f"   Sample type : {args.sample_type}")
    print(f"   Output dir  : {args.output_dir}")
    
    # Déterminer les fichiers
    if args.input:
        files = [args.input]
        print(f"   Mode        : Fichier unique")
    else:
        files = sorted(args.input_dir.glob("*.parquet"))
        files = [f for f in files if 'metadata' not in f.stem]
        print(f"   Mode        : Dossier ({len(files)} fichiers)")
        print(f"   Input dir   : {args.input_dir}")
    
    if not files:
        print(f"\n❌ Aucun fichier .parquet trouvé")
        return 1
    
    # Force mode
    if args.force:
        print(f"\n⚠️  Mode --force : écrasement des fichiers existants")
        for f in files:
            output_path = args.output_dir / f"{f.stem}_gt.json"
            if output_path.exists():
                output_path.unlink()
    
    # Traiter
    results = []
    total_start = time.time()
    
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] ", end="")
        result = process_single_file(f, args.output_dir, args.sample_type)
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
    total_bboxes = sum(r.get('n_bboxes', 0) for r in results)
    
    print(f"\n   Fichiers traités : {n_processed}")
    print(f"   Fichiers ignorés : {n_skipped}")
    print(f"   Temps total      : {total_elapsed:.1f}s")
    print(f"\n   📦 Total bboxes  : {total_bboxes}")
    
    # Résumé par fichier
    print(f"\n📊 Bboxes par scène :")
    for r in results:
        status = "✓" if r['status'] == 'processed' else "⏭"
        print(f"   {status} {r['file']}: {r['n_bboxes']} bboxes")
    
    # Comptage global par classe
    print(f"\n📊 Bboxes par classe (total) :")
    class_counts = {name: 0 for name in CLASS_NAMES.values()}
    
    for f in args.output_dir.glob("*_gt.json"):
        with open(f) as fp:
            data = json.load(fp)
            for bbox in data['bboxes']:
                class_counts[bbox['class_name']] += 1
    
    for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"   {name:15s}: {count}")
    
    print("\n" + "=" * 70)
    print("✅ TERMINÉ")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
