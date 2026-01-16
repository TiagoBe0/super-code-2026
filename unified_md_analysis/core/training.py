"""
Training: Entrenamiento de modelo Random Forest
Entrena un modelo de regresión para predecir número de vacancias
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from ..utils.constants import N_ESTIMATORS, RANDOM_STATE, TEST_SIZE, FEATURE_ORDER


class ModelTrainer:
    """
    Entrenador de modelo Random Forest para predicción de vacancias

    Pipeline:
    1. Carga dataset con features + target (n_vacancies)
    2. Divide en train/test
    3. Entrena Random Forest
    4. Evalúa métricas (RMSE, MAE, R²)
    5. Guarda modelo y metadatos
    """

    def __init__(self,
                 n_estimators: int = N_ESTIMATORS,
                 random_state: int = RANDOM_STATE,
                 test_size: float = TEST_SIZE):
        """
        Args:
            n_estimators: número de árboles en el bosque
            random_state: semilla para reproducibilidad
            test_size: proporción de datos para testing
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.test_size = test_size

        self.model = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred_train = None
        self.y_pred_test = None

    def load_data(self, csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Carga dataset desde CSV

        Args:
            csv_path: path al CSV con features + target

        Returns:
            X: DataFrame con features
            y: Series con target (n_vacancies)
        """
        print(f"📂 Cargando dataset: {csv_path}")
        df = pd.read_csv(csv_path)

        # Si hay columna 'file', usarla como índice
        if 'file' in df.columns:
            df = df.set_index('file')

        print(f"   Muestras: {df.shape[0]} | Columnas: {df.shape[1]}")

        # Verificar target
        if 'n_vacancies' not in df.columns:
            raise ValueError("❌ No se encontró 'n_vacancies' en el dataset")

        # Separar target (y) y features (X)
        y = df['n_vacancies'].astype(float)
        X = df.drop(columns=['n_vacancies'])

        # Solo columnas numéricas (por si hay alguna metadata)
        X = X.select_dtypes(include=[np.number])

        self.feature_names = list(X.columns)

        print(f"   Features de entrada: {len(X.columns)}")
        print(f"\n📊 Estadísticas de vacancias:")
        print(f"   Min: {y.min():.0f} | Max: {y.max():.0f} | Media: {y.mean():.1f}")

        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Entrena el modelo Random Forest

        Args:
            X: DataFrame con features
            y: Series con target
        """
        # Dividir en train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        print(f"\n🔀 División de datos:")
        print(f"   Entrenamiento: {len(self.X_train)} muestras")
        print(f"   Prueba: {len(self.X_test)} muestras")

        # Entrenar Random Forest
        print(f"\n🌲 Entrenando Random Forest...")

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )

        self.model.fit(self.X_train, self.y_train)
        print("   ✅ Entrenamiento completo")

        # Predicciones
        self.y_pred_train = self.model.predict(self.X_train)
        self.y_pred_test = self.model.predict(self.X_test)

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """
        Evalúa el modelo y retorna métricas

        Returns:
            metrics: dict con métricas de train y test
                - rmse: Root Mean Squared Error
                - mae: Mean Absolute Error
                - r2: R² Score
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Usa train() primero.")

        # Métricas train
        rmse_train = np.sqrt(mean_squared_error(self.y_train, self.y_pred_train))
        mae_train = mean_absolute_error(self.y_train, self.y_pred_train)
        r2_train = r2_score(self.y_train, self.y_pred_train)

        # Métricas test
        rmse_test = np.sqrt(mean_squared_error(self.y_test, self.y_pred_test))
        mae_test = mean_absolute_error(self.y_test, self.y_pred_test)
        r2_test = r2_score(self.y_test, self.y_pred_test)

        print("\n" + "="*60)
        print("📈 RESULTADOS DEL MODELO")
        print("="*60)

        print("\n🔵 ENTRENAMIENTO:")
        print(f"   RMSE: {rmse_train:.4f}")
        print(f"   MAE:  {mae_train:.4f}")
        print(f"   R²:   {r2_train:.4f}")

        print("\n🟢 PRUEBA:")
        print(f"   RMSE: {rmse_test:.4f}")
        print(f"   MAE:  {mae_test:.4f}")
        print(f"   R²:   {r2_test:.4f}")

        return {
            'train': {'rmse': rmse_train, 'mae': mae_train, 'r2': r2_train},
            'test': {'rmse': rmse_test, 'mae': mae_test, 'r2': r2_test}
        }

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Obtiene importancia de features

        Args:
            top_n: número de top features a retornar (None = todas)

        Returns:
            DataFrame con features e importancias ordenadas
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Usa train() primero.")

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        if top_n is not None:
            importance_df = importance_df.head(top_n)

        return importance_df

    def print_feature_importance(self, top_n: int = 15) -> None:
        """
        Imprime importancia de features

        Args:
            top_n: número de top features a mostrar
        """
        importance_df = self.get_feature_importance(top_n=top_n)

        print("\n" + "="*60)
        print(f"⭐ TOP {min(top_n, len(importance_df))} FEATURES MÁS IMPORTANTES")
        print("="*60)

        for idx, row in importance_df.iterrows():
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

            print(f"{emoji} {row['feature']:30s}: {row['importance']:.4f}")

    def save(self, output_dir: str, model_name: str = 'modelo_rf') -> None:
        """
        Guarda modelo y metadatos

        Args:
            output_dir: directorio de salida
            model_name: nombre base del modelo
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Usa train() primero.")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Guardar modelo
        model_path = Path(output_dir) / f'{model_name}.joblib'
        joblib.dump(self.model, model_path)
        print(f"\n💾 Modelo guardado: {model_path}")

        # Guardar nombres de features
        features_path = Path(output_dir) / 'feature_names.txt'
        with open(features_path, 'w') as f:
            for fname in self.feature_names:
                f.write(f"{fname}\n")
        print(f"📝 Features guardadas: {features_path}")

        # Guardar feature importance
        importance_df = self.get_feature_importance()
        importance_path = Path(output_dir) / 'feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        print(f"⭐ Importancias guardadas: {importance_path}")

    @staticmethod
    def load_model(model_path: str) -> RandomForestRegressor:
        """
        Carga modelo previamente guardado

        Args:
            model_path: path al archivo .joblib

        Returns:
            modelo cargado
        """
        return joblib.load(model_path)

    @staticmethod
    def load_feature_names(features_path: str) -> List[str]:
        """
        Carga nombres de features

        Args:
            features_path: path al archivo feature_names.txt

        Returns:
            lista de nombres de features
        """
        with open(features_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
