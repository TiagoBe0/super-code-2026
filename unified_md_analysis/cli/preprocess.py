#!/usr/bin/env python3
"""
CLI para preprocesamiento: Alpha Shape con Ghost Particles
"""

import argparse
import sys
from pathlib import Path

# Añadir parent directory al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unified_md_analysis.core.preprocessing import AlphaShapeWithGhosts
from unified_md_analysis.utils.lammps_parser import LAMMPSDumpParser
from unified_md_analysis.utils.constants import DEFAULT_PROBE_RADIUS


def main():
    parser = argparse.ArgumentParser(
        description="Preprocesamiento: Alpha Shape con Ghost Particles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Uso básico
  python preprocess.py input.dump output.dump

  # Con parámetros personalizados
  python preprocess.py input.dump output.dump --probe-radius 2.2 --num-ghost-layers 3

  # Con suavizado
  python preprocess.py input.dump output.dump --smoothing 10
        """
    )

    parser.add_argument("input_dump", help="Archivo dump de entrada")
    parser.add_argument("output_dump", help="Archivo dump de salida (solo superficie)")
    parser.add_argument("--probe-radius", type=float, default=DEFAULT_PROBE_RADIUS,
                        help=f"Radio de sonda en Angstroms (default: {DEFAULT_PROBE_RADIUS})")
    parser.add_argument("--lattice-param", type=float, default=None,
                        help="Parámetro de red (default: auto-detectar)")
    parser.add_argument("--num-ghost-layers", type=int, default=2,
                        help="Número de capas fantasma (default: 2)")
    parser.add_argument("--smoothing", type=int, default=0,
                        help="Iteraciones de suavizado Laplaciano (default: 0)")

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"{'PREPROCESAMIENTO: ALPHA SHAPE CON GHOST PARTICLES':^80}")
    print(f"{'='*80}\n")

    # Leer
    print(f"[CARGA] Leyendo: {args.input_dump}")
    data = LAMMPSDumpParser.read(args.input_dump)
    print(f"  ✓ Átomos: {len(data['atoms'])}")
    print(f"  ✓ Box: {data['box_bounds']}")

    # Procesar
    print(f"\n[PROCESAMIENTO]")
    constructor = AlphaShapeWithGhosts(
        positions=data['positions'],
        probe_radius=args.probe_radius,
        box_bounds=data['box_bounds'],
        lattice_param=args.lattice_param,
        num_ghost_layers=args.num_ghost_layers,
        smoothing_level=args.smoothing
    )

    constructor.perform()
    surface_atoms = constructor.get_surface_atoms_indices()

    # Exportar
    print(f"\n[EXPORTACIÓN] Escribiendo: {args.output_dump}")
    filtered_ids = sorted([data['atom_ids_ordered'][idx] for idx in surface_atoms])
    LAMMPSDumpParser.write(args.output_dump, data, filtered_ids)

    print(f"\n{'='*80}")
    print(f"✓ COMPLETADO")
    print(f"{'='*80}")
    print(f"  Entrada:  {len(data['atoms'])} átomos")
    print(f"  Salida:   {len(filtered_ids)} átomos")
    print(f"  Superficie: {constructor.surface_area:.2f} Ų")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
