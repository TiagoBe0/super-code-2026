"""
Prediction: Predicción de vacancias en nuevos dumps
Usa modelo entrenado para inferir número de vacancias
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional
from sklearn.ensemble import RandomForestRegressor

from .preprocessing import extract_all_features, features_to_array
from ..utils.lammps_parser import LAMMPSDumpParser
from ..utils.constants import FEATURE_ORDER


class VacancyPredictor:
    """
    Predictor de vacancias usando modelo Random Forest entrenado

    Pipeline:
    1. Carga modelo entrenado
    2. Lee dump file
    3. Extrae 35 features
    4. Predice número de vacancias
    """

    def __init__(self, model_path: str, feature_names_path: Optional[str] = None):
        """
        Args:
            model_path: path al modelo .joblib
            feature_names_path: path al archivo feature_names.txt (opcional)
        """
        self.model_path = model_path
        self.model = self._load_model(model_path)

        # Cargar nombres de features si se proporciona
        if feature_names_path:
            with open(feature_names_path, 'r') as f:
                self.feature_names = [line.strip() for line in f if line.strip()]
        else:
            self.feature_names = FEATURE_ORDER

        print(f"✅ Modelo cargado: {model_path}")
        print(f"   Features esperadas: {len(self.feature_names)}")

    def _load_model(self, model_path: str) -> RandomForestRegressor:
        """Carga modelo desde archivo"""
        import joblib
        return joblib.load(model_path)

    def predict_from_dump(self, dump_path: str) -> Dict:
        """
        Predice vacancias desde archivo dump

        Args:
            dump_path: path al archivo .dump

        Returns:
            result: dict con:
                - filename: nombre del archivo
                - n_atoms: número de átomos
                - predicted_vacancies: predicción
                - features: diccionario con features extraídas
        """
        print(f"\n📄 Procesando: {dump_path}")

        # Leer dump
        data = LAMMPSDumpParser.read(dump_path)
        positions = data['positions']
        n_atoms = len(positions)

        print(f"   Átomos: {n_atoms}")

        # Extraer features
        features_dict = extract_all_features(positions)
        features_array = features_to_array(features_dict)

        # Predecir
        prediction = self.model.predict(features_array.reshape(1, -1))[0]

        print(f"   ✅ Predicción: {prediction:.2f} vacancias")

        return {
            'filename': Path(dump_path).name,
            'n_atoms': n_atoms,
            'predicted_vacancies': float(prediction),
            'features': features_dict
        }

    def predict_from_positions(self, positions: np.ndarray, filename: str = 'unknown') -> Dict:
        """
        Predice vacancias desde array de posiciones

        Args:
            positions: array Nx3 de posiciones atómicas
            filename: nombre identificador (opcional)

        Returns:
            result: dict con predicción y features
        """
        n_atoms = len(positions)

        # Extraer features
        features_dict = extract_all_features(positions)
        features_array = features_to_array(features_dict)

        # Predecir
        prediction = self.model.predict(features_array.reshape(1, -1))[0]

        return {
            'filename': filename,
            'n_atoms': n_atoms,
            'predicted_vacancies': float(prediction),
            'features': features_dict
        }

    def predict_batch(self, dump_paths: List[str]) -> pd.DataFrame:
        """
        Predice vacancias para múltiples dumps

        Args:
            dump_paths: lista de paths a archivos .dump

        Returns:
            DataFrame con resultados
        """
        results = []

        print(f"\n🔄 Procesando {len(dump_paths)} archivos...")

        for i, dump_path in enumerate(dump_paths, 1):
            print(f"\n[{i}/{len(dump_paths)}]", end=" ")
            try:
                result = self.predict_from_dump(dump_path)
                results.append(result)
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({
                    'filename': Path(dump_path).name,
                    'n_atoms': np.nan,
                    'predicted_vacancies': np.nan,
                    'error': str(e)
                })

        # Convertir a DataFrame
        df_results = pd.DataFrame(results)

        # Reordenar columnas
        cols = ['filename', 'n_atoms', 'predicted_vacancies']
        if 'error' in df_results.columns:
            cols.append('error')
        df_results = df_results[cols]

        return df_results

    def predict_from_directory(self, directory: str, pattern: str = '*.dump') -> pd.DataFrame:
        """
        Predice vacancias para todos los dumps en un directorio

        Args:
            directory: path al directorio
            pattern: patrón de archivos (default: '*.dump')

        Returns:
            DataFrame con resultados
        """
        dump_paths = sorted(Path(directory).glob(pattern))
        dump_paths = [str(p) for p in dump_paths]

        if not dump_paths:
            raise ValueError(f"No se encontraron archivos con patrón '{pattern}' en {directory}")

        print(f"✅ Encontrados {len(dump_paths)} archivos")

        return self.predict_batch(dump_paths)

    def save_predictions(self, results: Union[pd.DataFrame, Dict], output_path: str) -> None:
        """
        Guarda predicciones a CSV

        Args:
            results: DataFrame o dict con resultados
            output_path: path al archivo CSV de salida
        """
        if isinstance(results, dict):
            results = pd.DataFrame([results])

        results.to_csv(output_path, index=False)
        print(f"\n💾 Resultados guardados: {output_path}")

    def summary_statistics(self, results: pd.DataFrame) -> None:
        """
        Imprime estadísticas de predicciones

        Args:
            results: DataFrame con predicciones
        """
        if 'predicted_vacancies' not in results.columns:
            print("⚠️ No hay columna 'predicted_vacancies' en resultados")
            return

        predictions = results['predicted_vacancies'].dropna()

        if len(predictions) == 0:
            print("⚠️ No hay predicciones válidas")
            return

        print("\n" + "="*60)
        print("📊 ESTADÍSTICAS DE PREDICCIONES")
        print("="*60)
        print(f"Total de archivos: {len(results)}")
        print(f"Predicciones exitosas: {len(predictions)}")
        print(f"\nVacancias predichas:")
        print(f"   Min:    {predictions.min():.2f}")
        print(f"   Max:    {predictions.max():.2f}")
        print(f"   Media:  {predictions.mean():.2f}")
        print(f"   Mediana: {predictions.median():.2f}")
        print(f"   Std:    {predictions.std():.2f}")

        # Mostrar archivos con errores si los hay
        if 'error' in results.columns:
            errors = results[results['error'].notna()]
            if len(errors) > 0:
                print(f"\n⚠️ Archivos con errores: {len(errors)}")
                for _, row in errors.iterrows():
                    print(f"   - {row['filename']}: {row['error']}")
