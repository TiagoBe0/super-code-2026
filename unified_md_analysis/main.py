#!/usr/bin/env python3
"""
Unified MD Analysis - Main Orchestrator
Interfaz unificada para análisis de estructuras FCC/BCC en dinámica molecular
"""

import argparse
import sys
from pathlib import Path


def create_parser():
    """Crea el parser de argumentos principal"""
    parser = argparse.ArgumentParser(
        description="Unified MD Analysis - Análisis de estructuras cristalinas FCC/BCC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PIPELINE COMPLETO:
  1. Alpha Shape: Extrae superficie con Ghost Particles
  2. Clustering (opcional): Separa nanoporos individuales
  3. Preprocesamiento: Calcula 37 features geométricas
  4. Training: Entrena modelo Random Forest
  5. Predicción: Predice vacancias en nuevos dumps

SUBCOMANDOS:
  alpha_shape   Detección de superficie con Alpha Shape
  cluster       Clustering de nanoporos
  preprocess    Extracción de features (preprocesamiento para ML)
  train         Entrenamiento de modelo
  predict       Predicción de vacancias

EJEMPLOS:
  # Ver ayuda de cada subcomando
  python main.py alpha_shape --help
  python main.py preprocess --help
  python main.py train --help

  # Pipeline completo (ejemplo)
  python main.py alpha_shape raw.dump surface.dump
  python main.py preprocess surface_dumps/ --output features.csv
  python main.py train features.csv --output models/
  python main.py predict models/modelo_rf.joblib new_dump.dump
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Subcomandos disponibles')

    # Cada subcomando delega a su CLI específico
    subparsers.add_parser('alpha_shape', add_help=False)
    subparsers.add_parser('cluster', add_help=False)
    subparsers.add_parser('preprocess', add_help=False)
    subparsers.add_parser('train', add_help=False)
    subparsers.add_parser('predict', add_help=False)

    return parser


def main():
    parser = create_parser()
    args, remaining_args = parser.parse_known_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Mapear comandos a CLIs
    cli_map = {
        'alpha_shape': 'cli.alpha_shape',
        'cluster': 'cli.cluster',
        'preprocess': 'cli.preprocess',
        'train': 'cli.train',
        'predict': 'cli.predict'
    }

    if args.command in cli_map:
        # Importar y ejecutar el CLI correspondiente
        module_name = cli_map[args.command]
        module = __import__(module_name, fromlist=['main'])

        # Restaurar sys.argv para el subcomando
        sys.argv = [f'{args.command}.py'] + remaining_args

        # Ejecutar
        module.main()
    else:
        print(f"❌ Comando desconocido: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
