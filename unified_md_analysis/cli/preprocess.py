#!/usr/bin/env python3
"""
CLI para preprocesamiento: Extracción de features para ML
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Añadir parent directory al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unified_md_analysis.core.preprocessing import extract_all_features
from unified_md_analysis.utils.lammps_parser import LAMMPSDumpParser


def main():
    parser = argparse.ArgumentParser(
        description="Preprocesamiento: Extracción de features para Machine Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Extrae 35 features geométricas de archivos dump:
  - 26 features del grid 3D de ocupación
  - 2 features del Convex Hull
  - 3 momentos de inercia principales
  - 2 features radiales (RDF)
  - 1 entropía espacial
  - 1 bandwidth de clustering

Ejemplos:
  # Extraer features de un directorio de dumps
  python preprocess.py dumps/ --output features.csv

  # Con target conocido (n_vacancies)
  python preprocess.py dumps/ --output features.csv --vacancies-file vacancies.txt
        """
    )

    parser.add_argument("input_dir", help="Directorio con archivos .dump")
    parser.add_argument("--output", "-o", required=True, help="Archivo CSV de salida")
    parser.add_argument("--pattern", default="*.dump", help="Patrón de archivos (default: *.dump)")
    parser.add_argument("--vacancies-file", help="Archivo con n_vacancies por dump (opcional)")

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"{'PREPROCESAMIENTO: EXTRACCIÓN DE FEATURES':^80}")
    print(f"{'='*80}\n")

    # Buscar dumps
    dump_paths = sorted(Path(args.input_dir).glob(args.pattern))
    if not dump_paths:
        print(f"❌ No se encontraron archivos con patrón '{args.pattern}' en {args.input_dir}")
        sys.exit(1)

    print(f"✅ Encontrados {len(dump_paths)} archivos")

    # Cargar n_vacancies si se proporciona
    vacancies_dict = {}
    if args.vacancies_file:
        print(f"📄 Cargando vacancias desde: {args.vacancies_file}")
        with open(args.vacancies_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    vacancies_dict[parts[0]] = int(parts[1])
        print(f"  ✓ {len(vacancies_dict)} vacancias cargadas")

    # Extraer features
    print(f"\n[EXTRACCIÓN] Procesando archivos...")
    results = []

    for i, dump_path in enumerate(dump_paths, 1):
        try:
            print(f"  [{i}/{len(dump_paths)}] {dump_path.name}...", end=" ")

            # Leer dump
            data = LAMMPSDumpParser.read(str(dump_path))
            positions = data['positions']

            # Extraer features
            features_dict = extract_all_features(positions)

            # Añadir metadata
            features_dict['file'] = dump_path.name
            features_dict['n_atoms'] = len(positions)

            # Añadir target si está disponible
            if dump_path.name in vacancies_dict:
                features_dict['n_vacancies'] = vacancies_dict[dump_path.name]

            results.append(features_dict)
            print("✓")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Crear DataFrame
    df = pd.DataFrame(results)

    # Reordenar columnas (file primero, luego features, luego n_vacancies si existe)
    cols = ['file', 'n_atoms']
    feature_cols = [c for c in df.columns if c not in ['file', 'n_atoms', 'n_vacancies']]
    cols.extend(feature_cols)
    if 'n_vacancies' in df.columns:
        cols.append('n_vacancies')

    df = df[cols]

    # Guardar
    df.to_csv(args.output, index=False)

    print(f"\n{'='*80}")
    print(f"✓ COMPLETADO")
    print(f"{'='*80}")
    print(f"  Archivos procesados: {len(results)}")
    print(f"  Features extraídas: {len(feature_cols)}")
    print(f"  Salida guardada: {args.output}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
