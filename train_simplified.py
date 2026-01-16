#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENTRENAMIENTO CON FEATURES SIMPLIFICADOS
Entrena Random Forest con las 15 features más importantes (99.8% de importancia)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(csv_path):
    """Carga el dataset y separa features del target"""
    logger.info(f"📂 Cargando dataset: {csv_path}")
    df = pd.read_csv(csv_path, index_col='file')
    
    logger.info(f"   Muestras: {df.shape[0]} | Columnas: {df.shape[1]}")
    
    # Verificar target
    if 'n_vacancies' not in df.columns:
        raise ValueError("❌ No se encontró 'n_vacancies' en el dataset")
    
    # Separar target (y) y features (X)
    y = df['n_vacancies'].astype(float)
    X = df.drop(columns=['n_vacancies'])
    
    # Solo columnas numéricas (por si hay alguna metadata)
    X = X.select_dtypes(include=[np.number])
    
    logger.info(f"   Features de entrada: {len(X.columns)}")
    logger.info(f"\n📊 Estadísticas de vacancias:")
    logger.info(f"   Min: {y.min():.0f} | Max: {y.max():.0f} | Media: {y.mean():.1f}")
    
    return X, y


def train_model(X, y, test_size=0.2, random_state=42):
    """Entrena el modelo Random Forest"""
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"\n🔀 División de datos:")
    logger.info(f"   Entrenamiento: {len(X_train)} muestras")
    logger.info(f"   Prueba: {len(X_test)} muestras")
    
    # Entrenar Random Forest
    logger.info(f"\n🌲 Entrenando Random Forest...")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    logger.info("   ✅ Entrenamiento completo")
    
    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'feature_names': list(X.columns)
    }


def evaluate_model(results):
    """Evalúa el modelo y muestra métricas"""
    y_train = results['y_train']
    y_test = results['y_test']
    y_pred_train = results['y_pred_train']
    y_pred_test = results['y_pred_test']
    
    # Métricas train
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    # Métricas test
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    logger.info("\n" + "="*60)
    logger.info("📈 RESULTADOS DEL MODELO")
    logger.info("="*60)
    
    logger.info("\n🔵 ENTRENAMIENTO:")
    logger.info(f"   RMSE: {rmse_train:.4f}")
    logger.info(f"   MAE:  {mae_train:.4f}")
    logger.info(f"   R²:   {r2_train:.4f}")
    
    logger.info("\n🟢 PRUEBA:")
    logger.info(f"   RMSE: {rmse_test:.4f}")
    logger.info(f"   MAE:  {mae_test:.4f}")
    logger.info(f"   R²:   {r2_test:.4f}")
    
    return {
        'train': {'rmse': rmse_train, 'mae': mae_train, 'r2': r2_train},
        'test': {'rmse': rmse_test, 'mae': mae_test, 'r2': r2_test}
    }


def show_feature_importance(results, top_n=15):
    """Muestra la importancia de cada feature"""
    model = results['model']
    feature_names = results['feature_names']
    
    # Crear DataFrame con importancias
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\n" + "="*60)
    logger.info(f"⭐ TOP {min(top_n, len(importance_df))} FEATURES MÁS IMPORTANTES")
    logger.info("="*60)
    
    for idx, row in importance_df.head(top_n).iterrows():
        # Emoji según tipo de feature
        if 'occupancy' in row['feature']:
            emoji = "📊"
        elif 'hull' in row['feature']:
            emoji = "🔷"
        elif 'moi' in row['feature']:
            emoji = "⚖️"
        elif 'rdf' in row['feature']:
            emoji = "📏"
        elif 'entropy' in row['feature']:
            emoji = "🌀"
        elif 'ms' in row['feature']:
            emoji = "🔍"
        else:
            emoji = "  "
        
        logger.info(f"{emoji} {row['feature']:30s}: {row['importance']:.4f}")
    
    return importance_df


def create_plots(results, metrics, output_dir):
    """Crea gráficos de evaluación"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    y_test = results['y_test']
    y_pred_test = results['y_pred_test']
    r2_test = metrics['test']['r2']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Evaluación del Modelo', fontsize=16, fontweight='bold')
    
    # 1. Predicciones vs Reales
    axes[0, 0].scatter(y_test, y_pred_test, alpha=0.6, s=50)
    axes[0, 0].plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Ideal')
    axes[0, 0].set_xlabel('Vacancias Reales', fontsize=12)
    axes[0, 0].set_ylabel('Vacancias Predichas', fontsize=12)
    axes[0, 0].set_title(f'Predicciones vs Reales\nR² = {r2_test:.4f}', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuos
    residuos = y_test - y_pred_test
    axes[0, 1].scatter(y_pred_test, residuos, alpha=0.6, s=50)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicciones', fontsize=12)
    axes[0, 1].set_ylabel('Residuos (Real - Predicción)', fontsize=12)
    axes[0, 1].set_title('Análisis de Residuos', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature Importance
    model = results['model']
    feature_names = results['feature_names']
    importances = model.feature_importances_
    
    # Ordenar
    indices = np.argsort(importances)[::-1][:15]  # Top 15
    
    axes[1, 0].barh(range(len(indices)), importances[indices])
    axes[1, 0].set_yticks(range(len(indices)))
    axes[1, 0].set_yticklabels([feature_names[i] for i in indices])
    axes[1, 0].set_xlabel('Importancia', fontsize=12)
    axes[1, 0].set_title('Top 15 Features', fontsize=12)
    axes[1, 0].invert_yaxis()
    
    # 4. Distribución de Errores
    axes[1, 1].hist(residuos, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Error (Real - Predicción)', fontsize=12)
    axes[1, 1].set_ylabel('Frecuencia', fontsize=12)
    axes[1, 1].set_title('Distribución de Errores', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_path = Path(output_dir) / 'evaluacion_modelo.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"\n📊 Gráficos guardados: {plot_path}")
    
    plt.close()


def save_model(model, feature_names, output_dir):
    """Guarda el modelo y sus metadatos"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Guardar modelo
    model_path = Path(output_dir) / 'modelo_rf_simplificado.pkl'
    joblib.dump(model, model_path)
    logger.info(f"💾 Modelo guardado: {model_path}")
    
    # Guardar nombres de features
    features_path = Path(output_dir) / 'feature_names.txt'
    with open(features_path, 'w') as f:
        for fname in feature_names:
            f.write(f"{fname}\n")
    
    logger.info(f"📝 Features guardadas: {features_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Entrena Random Forest con features simplificados",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python train_simplified.py \\
      -i simplified_features/dataset_top_features.csv \\
      -o modelo_simplificado \\
      --test-size 0.2

El modelo se entrena con 15 features que representan 99.8% de importancia.
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='CSV con dataset de features')
    parser.add_argument('-o', '--output', default='modelo_simplificado',
                       help='Directorio de salida')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proporción para test (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Semilla aleatoria')
    
    args = parser.parse_args()
    
    # Validar archivo
    if not Path(args.input).exists():
        logger.error(f"❌ Archivo no encontrado: {args.input}")
        return
    
    try:
        logger.info("="*70)
        logger.info("🚀 ENTRENAMIENTO DE MODELO SIMPLIFICADO")
        logger.info("="*70)
        
        # 1. Cargar datos
        X, y = load_data(args.input)
        
        # 2. Entrenar
        results = train_model(X, y, args.test_size, args.random_state)
        
        # 3. Evaluar
        metrics = evaluate_model(results)
        
        # 4. Feature importance
        importance_df = show_feature_importance(results)
        
        # 5. Crear gráficos
        create_plots(results, metrics, args.output)
        
        # 6. Guardar modelo
        save_model(results['model'], results['feature_names'], args.output)
        
        # 7. Guardar importancias
        importance_path = Path(args.output) / 'feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"📋 Importancias guardadas: {importance_path}")
        
        logger.info("\n" + "="*70)
        logger.info("✅ ENTRENAMIENTO COMPLETADO")
        logger.info("="*70)
        logger.info(f"\n📁 Archivos generados en: {args.output}/")
        logger.info(f"   • modelo_rf_simplificado.pkl")
        logger.info(f"   • feature_names.txt")
        logger.info(f"   • feature_importance.csv")
        logger.info(f"   • evaluacion_modelo.png")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


    """
    
        python train_simplified.py \
    -i features_enhanced/dataset_enhanced_features.csv \
    -o mi_modelo
    """