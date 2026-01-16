#!/usr/bin/env python3
"""
CLI para clustering de nanoporos
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Añadir parent directory al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unified_md_analysis.core.clustering import ClusteringEngine, HDBSCAN_AVAILABLE
from unified_md_analysis.utils.lammps_parser import LAMMPSDumpParser
from unified_md_analysis.utils.constants import DEFAULT_MIN_CLUSTER_SIZE, DEFAULT_MIN_SAMPLES


def main():
    parser = argparse.ArgumentParser(
        description="Clustering de nanoporos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Algoritmos disponibles:
  - hdbscan: Clustering jerárquico basado en densidad (detecta ruido)
  - kmeans: Particionamiento en K clusters
  - meanshift: Basado en densidad con estimación automática
  - agglomerative: Clustering jerárquico aglomerativo

Ejemplos:
  # HDBSCAN (automático)
  python cluster.py input.dump output_dir --method hdbscan

  # KMeans con 5 clusters
  python cluster.py input.dump output_dir --method kmeans --n-clusters 5

  # MeanShift
  python cluster.py input.dump output_dir --method meanshift
        """
    )

    parser.add_argument("input_dump", help="Archivo dump de entrada")
    parser.add_argument("output_dir", help="Directorio de salida para clusters")
    parser.add_argument("--method", choices=['hdbscan', 'kmeans', 'meanshift', 'agglomerative'],
                        default='hdbscan', help="Algoritmo de clustering")
    parser.add_argument("--n-clusters", type=int, default=5,
                        help="Número de clusters (kmeans/agglomerative)")
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE,
                        help=f"Tamaño mínimo de cluster (hdbscan) (default: {DEFAULT_MIN_CLUSTER_SIZE})")
    parser.add_argument("--quantile", type=float, default=0.2,
                        help="Quantile para bandwidth (meanshift) (default: 0.2)")

    args = parser.parse_args()

    # Validar HDBSCAN
    if args.method == 'hdbscan' and not HDBSCAN_AVAILABLE:
        print("❌ HDBSCAN no está instalado. Instala con: pip install hdbscan")
        print("   O usa otro método: --method kmeans")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"{'CLUSTERING DE NANOPOROS':^80}")
    print(f"{'='*80}\n")

    # Leer
    print(f"[CARGA] Leyendo: {args.input_dump}")
    data = LAMMPSDumpParser.read(args.input_dump)
    positions = data['positions']
    print(f"  ✓ Átomos: {len(positions)}")

    # Clustering
    print(f"\n[CLUSTERING] Método: {args.method.upper()}")
    engine = ClusteringEngine(positions)

    if args.method == 'hdbscan':
        n_clusters = engine.apply_hdbscan(min_cluster_size=args.min_cluster_size)
    elif args.method == 'kmeans':
        n_clusters = engine.apply_kmeans(n_clusters=args.n_clusters)
    elif args.method == 'meanshift':
        n_clusters = engine.apply_meanshift(quantile=args.quantile)
    elif args.method == 'agglomerative':
        n_clusters = engine.apply_agglomerative(n_clusters=args.n_clusters)

    print(f"  ✓ Clusters detectados: {n_clusters}")

    # Mostrar resumen
    print("\n" + engine.summary())

    # Crear directorio de salida
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dividir por clusters y guardar
    clusters_dict = engine.split_by_clusters(positions)

    print(f"\n[EXPORTACIÓN] Guardando en: {args.output_dir}")

    for cluster_id, cluster_positions in clusters_dict.items():
        if cluster_id == -1:
            filename = "noise.dump"
        else:
            filename = f"cluster_{cluster_id}.dump"

        output_file = output_path / filename
        LAMMPSDumpParser.write_simple(
            str(output_file),
            cluster_positions,
            timestep=data['timestep'],
            box_bounds=data['box_bounds']
        )
        print(f"  ✓ {filename}: {len(cluster_positions)} átomos")

    print(f"\n{'='*80}")
    print(f"✓ COMPLETADO")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
