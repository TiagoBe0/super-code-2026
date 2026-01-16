#!/usr/bin/env python3
"""
CLI para predicción de vacancias
"""

import argparse
import sys
from pathlib import Path

# Añadir parent directory al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unified_md_analysis.core.prediction import VacancyPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Predicción de vacancias usando modelo entrenado",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Predice número de vacancias en dumps usando modelo Random Forest.

Ejemplos:
  # Predecir un archivo
  python predict.py modelo.joblib input.dump

  # Predecir múltiples archivos
  python predict.py modelo.joblib dumps/ --output predictions.csv

  # Especificar patrón de archivos
  python predict.py modelo.joblib dumps/ --pattern "*.dump" --output results.csv
        """
    )

    parser.add_argument("model_path", help="Path al modelo .joblib")
    parser.add_argument("input", help="Archivo .dump o directorio con dumps")
    parser.add_argument("--output", "-o", help="CSV de salida (opcional)")
    parser.add_argument("--pattern", default="*.dump", help="Patrón de archivos (default: *.dump)")
    parser.add_argument("--features", help="Path a feature_names.txt (opcional)")

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"{'PREDICCIÓN DE VACANCIAS':^80}")
    print(f"{'='*80}\n")

    # Crear predictor
    predictor = VacancyPredictor(args.model_path, args.features)

    # Determinar si es archivo o directorio
    input_path = Path(args.input)

    if input_path.is_file():
        # Un solo archivo
        result = predictor.predict_from_dump(str(input_path))

        print(f"\n{'='*80}")
        print(f"✓ PREDICCIÓN COMPLETADA")
        print(f"{'='*80}")
        print(f"  Archivo: {result['filename']}")
        print(f"  Átomos: {result['n_atoms']}")
        print(f"  Vacancias predichas: {result['predicted_vacancies']:.2f}")
        print(f"{'='*80}\n")

        if args.output:
            predictor.save_predictions(result, args.output)

    elif input_path.is_dir():
        # Directorio
        results = predictor.predict_from_directory(str(input_path), args.pattern)

        # Mostrar estadísticas
        predictor.summary_statistics(results)

        # Guardar si se especifica
        if args.output:
            predictor.save_predictions(results, args.output)
        else:
            print("\n💡 Tip: Usa --output para guardar predicciones a CSV")

    else:
        print(f"❌ Error: '{args.input}' no es un archivo ni directorio válido")
        sys.exit(1)


if __name__ == "__main__":
    main()
