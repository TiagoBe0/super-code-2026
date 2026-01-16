#!/usr/bin/env python3
"""
CLI para entrenamiento del modelo
"""

import argparse
import sys
from pathlib import Path

# Añadir parent directory al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unified_md_analysis.core.training import ModelTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelo Random Forest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Entrena un modelo Random Forest para predecir número de vacancias.

El CSV de entrada debe contener:
  - Columnas de features (35 features geométricas)
  - Columna 'n_vacancies' (target)

Ejemplos:
  # Entrenamiento básico
  python train.py features.csv --output models/

  # Con parámetros personalizados
  python train.py features.csv --output models/ --n-estimators 200 --test-size 0.3
        """
    )

    parser.add_argument("input_csv", help="CSV con features y target (n_vacancies)")
    parser.add_argument("--output", "-o", default="models", help="Directorio de salida")
    parser.add_argument("--model-name", default="modelo_rf", help="Nombre del modelo")
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Número de árboles (default: 100)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proporción de test (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Semilla aleatoria (default: 42)")

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"{'ENTRENAMIENTO DE MODELO':^80}")
    print(f"{'='*80}\n")

    # Crear trainer
    trainer = ModelTrainer(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        test_size=args.test_size
    )

    # Cargar datos
    X, y = trainer.load_data(args.input_csv)

    # Entrenar
    trainer.train(X, y)

    # Evaluar
    metrics = trainer.evaluate()

    # Mostrar importancia de features
    trainer.print_feature_importance(top_n=15)

    # Guardar
    trainer.save(args.output, model_name=args.model_name)

    print(f"\n{'='*80}")
    print(f"✓ ENTRENAMIENTO COMPLETADO")
    print(f"{'='*80}")
    print(f"  R² (test): {metrics['test']['r2']:.4f}")
    print(f"  RMSE (test): {metrics['test']['rmse']:.4f}")
    print(f"  Modelo: {Path(args.output) / f'{args.model_name}.joblib'}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
