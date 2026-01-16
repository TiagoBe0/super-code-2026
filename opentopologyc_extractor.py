#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXTRACTOR FINAL OPTIMIZADO - VERSIÓN PRODUCCIÓN
Calcula SOLO las 20 features finales especificadas
Centrado correcto del origen de coordenadas ANTES de calcular grid

FEATURES FINALES (20):
1. occupancy_total
2. occupancy_fraction
3. occupancy_x_mean, occupancy_y_mean, occupancy_z_mean
4. occupancy_gradient_x, occupancy_gradient_y, occupancy_gradient_z, occupancy_gradient_total
5. occupancy_surface
6. grid_entropy
7. grid_moi_1, grid_moi_2, grid_moi_3
8. moi_principal_3
9. rdf_mean, rdf_kurtosis
10. entropy_spatial
11. ms_bandwidth

VALIDACIONES:
✓ Centrado de coordenadas ANTES del grid (origen en centro de masa)
✓ PCA optimizado con covariance_eigh
✓ Cálculo correcto de momentos de inercia
✓ Manejo de archivos sin errores
✓ Logs detallados de cada paso
"""

import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')

import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import ConvexHull
from scipy.stats import kurtosis, entropy
from scipy.ndimage import label
from sklearn.decomposition import PCA
from sklearn.cluster import estimate_bandwidth
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OVITO
try:
    from ovito.io import import_file
    from ovito.modifiers import (
        ConstructSurfaceModifier,
        InvertSelectionModifier,
        DeleteSelectedModifier
    )
except ImportError as e:
    logger.error(f"❌ Error importando OVITO: {e}")
    logger.error("   Instala con: pip install ovito")
    sys.exit(1)

# ==================== CONFIGURACIÓN ====================
ATM_TOTAL = 16384
A0 = 3.532
GRID_SIZE = (10, 10, 10)
BOX_SIZE_MAX = 10.0

# Features finales a calcular
FINAL_FEATURES = [
    'occupancy_total',
    'occupancy_fraction',
    'occupancy_x_mean',
    'occupancy_y_mean',
    'occupancy_z_mean',
    'occupancy_gradient_x',
    'occupancy_gradient_y',
    'occupancy_gradient_z',
    'occupancy_gradient_total',
    'occupancy_surface',
    'grid_entropy',
    'grid_moi_1',
    'grid_moi_2',
    'grid_moi_3',
    'moi_principal_3',
    'rdf_mean',
    'rdf_kurtosis',
    'entropy_spatial',
    'ms_bandwidth',
    'n_vacancies'
]


# ==================== EXTRACCIÓN DE SUPERFICIE ====================

def extract_surface_atoms(dump_path):
    """
    Extrae átomos de superficie usando OVITO
    
    Returns:
        positions: array (N, 3) de posiciones de superficie
        n_vacancies: número de vacancias detectadas
    """
    try:
        pipeline = import_file(dump_path)
        
        # Detectar superficie
        pipeline.modifiers.append(ConstructSurfaceModifier(
            radius=2.0,
            select_surface_particles=True,
            smoothing_level=12,
            compute_distances=True
        ))
        
        # Contar vacancias antes de eliminar
        data_full = pipeline.compute()
        n_vacancies = ATM_TOTAL - data_full.particles.count
        
        # Invertir selección y eliminar no-superficie
        pipeline.modifiers.append(InvertSelectionModifier())
        pipeline.modifiers.append(DeleteSelectedModifier())
        
        # Obtener posiciones de superficie
        data_surface = pipeline.compute()
        positions = data_surface.particles['Position'][...]
        
        logger.info(f"   ✓ Superficie: {len(positions)} átomos")
        logger.info(f"   ✓ Vacancias: {n_vacancies}")
        
        return positions, n_vacancies
        
    except Exception as e:
        logger.error(f"   ❌ Error extrayendo superficie: {e}")
        raise


# ==================== NORMALIZACIÓN Y CENTRADO ====================

def normalize_positions(positions):
    """
    Normaliza y alinea posiciones usando PCA optimizado
    
    IMPORTANTE: 
    - Centra en origin (mean = 0) ANTES de calcular PCA
    - Usa covariance_eigh para n_samples >> n_features
    - Retorna posiciones centradas y normalizadas
    
    Args:
        positions: array (N, 3) de coordenadas
        
    Returns:
        normalized: array (N, 3) centrado y normalizado
        box_size: tamaño de caja para grid
    """
    if len(positions) < 3:
        logger.warning(f"   ⚠️  Pocas posiciones ({len(positions)}), usando centering simple")
        centered = positions - positions.mean(axis=0)
        return centered / A0, 2.0
    
    # PASO 1: Centrar en origin
    centered = positions - positions.mean(axis=0)
    
    # PASO 2: PCA optimizado
    try:
        pca = PCA(n_components=3, svd_solver='covariance_eigh')
        aligned = pca.fit_transform(centered)
        logger.info(f"   ✓ PCA optimizado (covariance_eigh)")
    except Exception as e:
        logger.warning(f"   ⚠️  Fallback a PCA 'auto': {e}")
        try:
            pca = PCA(n_components=3, svd_solver='auto')
            aligned = pca.fit_transform(centered)
        except Exception as e2:
            logger.error(f"   ❌ PCA fallido: {e2}")
            raise
    
    # PASO 3: Normalizar por A0
    normalized = aligned / A0
    
    # PASO 4: Calcular tamaño de caja
    extent = normalized.max(axis=0) - normalized.min(axis=0)
    box_size = max(1.5, min(extent.max() * 1.5, BOX_SIZE_MAX))
    
    logger.info(f"   ✓ Extent: {extent}")
    logger.info(f"   ✓ Box size: {box_size:.2f}")
    
    return normalized, box_size


# ==================== CÁLCULO DE FEATURES ====================

def calc_grid_features(positions, box_size):
    """
    Calcula features del grid 3D SOLO las finales
    
    IMPORTANTE: Las posiciones DEBEN estar centradas antes de llamar
    
    Returns:
        dict con features del grid
    """
    N, M, L = GRID_SIZE
    grid = np.zeros((N, M, L), dtype=np.int8)
    
    half_box = box_size / 2.0
    cell_size = box_size / np.array(GRID_SIZE)
    
    # ========== LLENAR GRID ==========
    for pos in positions:
        indices = np.floor((pos + half_box) / cell_size).astype(int)
        if np.all(indices >= 0) and np.all(indices < GRID_SIZE):
            grid[indices[0], indices[1], indices[2]] = 1
    
    occupancy_total = int(grid.sum())
    features = {}
    
    # ========== 1. OCCUPANCY BÁSICAS ==========
    features['occupancy_total'] = float(occupancy_total)
    features['occupancy_fraction'] = float(grid.mean())
    
    # Media de ocupación por eje
    for axis, name in enumerate(['x', 'y', 'z']):
        slices = grid.sum(axis=axis)
        features[f'occupancy_{name}_mean'] = float(slices.mean())
    
    # ========== 2. GRADIENTES ==========
    if occupancy_total > 0:
        grad_x = np.abs(np.diff(grid, axis=0)).sum()
        grad_y = np.abs(np.diff(grid, axis=1)).sum()
        grad_z = np.abs(np.diff(grid, axis=2)).sum()
        
        features['occupancy_gradient_x'] = float(grad_x)
        features['occupancy_gradient_y'] = float(grad_y)
        features['occupancy_gradient_z'] = float(grad_z)
        features['occupancy_gradient_total'] = float(grad_x + grad_y + grad_z)
    else:
        features['occupancy_gradient_x'] = 0.0
        features['occupancy_gradient_y'] = 0.0
        features['occupancy_gradient_z'] = 0.0
        features['occupancy_gradient_total'] = 0.0
    
    # ========== 3. SUPERFICIE ==========
    if occupancy_total > 0:
        surface = (np.abs(np.diff(grid, axis=0)).sum() +
                   np.abs(np.diff(grid, axis=1)).sum() +
                   np.abs(np.diff(grid, axis=2)).sum())
        features['occupancy_surface'] = float(surface)
    else:
        features['occupancy_surface'] = 0.0
    
    # ========== 4. ENTROPÍA ==========
    if occupancy_total > 0:
        prob = grid.flatten()
        prob = prob[prob > 0] / occupancy_total
        grid_entropy = -np.sum(prob * np.log(prob + 1e-10))
        features['grid_entropy'] = float(grid_entropy)
    else:
        features['grid_entropy'] = 0.0
    
    # ========== 5. MOMENTOS DE INERCIA DEL GRID ==========
    if occupancy_total > 0:
        try:
            coords = np.argwhere(grid == 1)
            centered_grid = coords - coords.mean(axis=0)
            
            Ixx = np.sum(centered_grid[:, 1]**2 + centered_grid[:, 2]**2)
            Iyy = np.sum(centered_grid[:, 0]**2 + centered_grid[:, 2]**2)
            Izz = np.sum(centered_grid[:, 0]**2 + centered_grid[:, 1]**2)
            Ixy = -np.sum(centered_grid[:, 0] * centered_grid[:, 1])
            Ixz = -np.sum(centered_grid[:, 0] * centered_grid[:, 2])
            Iyz = -np.sum(centered_grid[:, 1] * centered_grid[:, 2])
            
            I_tensor = np.array([
                [Ixx, Ixy, Ixz],
                [Ixy, Iyy, Iyz],
                [Ixz, Iyz, Izz]
            ])
            
            eigenvalues = np.sort(np.linalg.eigvalsh(I_tensor))[::-1]
            features['grid_moi_1'] = float(eigenvalues[0])
            features['grid_moi_2'] = float(eigenvalues[1])
            features['grid_moi_3'] = float(eigenvalues[2])
        except Exception as e:
            logger.warning(f"   ⚠️  Error en MOI del grid: {e}")
            features['grid_moi_1'] = 0.0
            features['grid_moi_2'] = 0.0
            features['grid_moi_3'] = 0.0
    else:
        features['grid_moi_1'] = 0.0
        features['grid_moi_2'] = 0.0
        features['grid_moi_3'] = 0.0
    
    logger.info(f"   ✓ Grid features: {len([k for k in features.keys()])} features")
    
    return features


def calc_inertia_features(positions):
    """
    Calcula SOLO moi_principal_3 (requiere moi_principal_1, 2, 3)
    
    Returns:
        dict con {'moi_principal_3': valor}
    """
    if len(positions) < 3:
        logger.warning(f"   ⚠️  Pocas posiciones para MOI ({len(positions)})")
        return {'moi_principal_3': np.nan}
    
    try:
        centered = positions - positions.mean(axis=0)
        
        Ixx = np.sum(centered[:, 1]**2 + centered[:, 2]**2)
        Iyy = np.sum(centered[:, 0]**2 + centered[:, 2]**2)
        Izz = np.sum(centered[:, 0]**2 + centered[:, 1]**2)
        
        Ixy = -np.sum(centered[:, 0] * centered[:, 1])
        Ixz = -np.sum(centered[:, 0] * centered[:, 2])
        Iyz = -np.sum(centered[:, 1] * centered[:, 2])
        
        I_tensor = np.array([
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz]
        ])
        
        eigenvalues = np.sort(np.linalg.eigvalsh(I_tensor))[::-1]
        
        return {'moi_principal_3': float(eigenvalues[2])}
    except Exception as e:
        logger.warning(f"   ⚠️  Error en MOI principal: {e}")
        return {'moi_principal_3': np.nan}


def calc_radial_features(positions):
    """
    Calcula rdf_mean y rdf_kurtosis
    """
    if len(positions) < 2:
        logger.warning(f"   ⚠️  Pocas posiciones para RDF ({len(positions)})")
        return {'rdf_mean': np.nan, 'rdf_kurtosis': np.nan}
    
    try:
        com = positions.mean(axis=0)
        distances = np.linalg.norm(positions - com, axis=1)
        
        return {
            'rdf_mean': float(distances.mean()),
            'rdf_kurtosis': float(kurtosis(distances))
        }
    except Exception as e:
        logger.warning(f"   ⚠️  Error en RDF: {e}")
        return {'rdf_mean': np.nan, 'rdf_kurtosis': np.nan}


def calc_entropy_feature(positions):
    """
    Calcula entropy_spatial
    """
    if len(positions) < 2:
        logger.warning(f"   ⚠️  Pocas posiciones para entropía ({len(positions)})")
        return {'entropy_spatial': np.nan}
    
    try:
        H, _ = np.histogramdd(positions, bins=10)
        H_flat = H.flatten()
        H_norm = H_flat[H_flat > 0] / H_flat.sum()
        
        return {'entropy_spatial': float(entropy(H_norm))}
    except Exception as e:
        logger.warning(f"   ⚠️  Error en entropía espacial: {e}")
        return {'entropy_spatial': np.nan}


def calc_bandwidth_feature(positions):
    """
    Calcula ms_bandwidth (Mean Shift bandwidth)
    """
    if len(positions) < 10:
        logger.warning(f"   ⚠️  Pocas posiciones para bandwidth ({len(positions)})")
        return {'ms_bandwidth': np.nan}
    
    try:
        bandwidth = estimate_bandwidth(
            positions,
            quantile=0.2,
            n_samples=min(500, len(positions))
        )
        
        if bandwidth <= 0:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(positions)
            distances, _ = nn.kneighbors(positions)
            bandwidth = np.mean(distances[:, 1]) * 2.0
        
        return {'ms_bandwidth': float(bandwidth)}
    except Exception as e:
        logger.warning(f"   ⚠️  Error en bandwidth: {e}")
        return {'ms_bandwidth': np.nan}


# ==================== PIPELINE COMPLETO ====================

def extract_features_optimized(dump_path):
    """
    Pipeline COMPLETO y VALIDADO
    
    ORDEN DE OPERACIONES:
    1. Extraer superficie
    2. Centrar coordenadas (IMPORTANTE)
    3. PCA normalización
    4. Calcular todas las features finales
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 Procesando: {Path(dump_path).name}")
    logger.info(f"{'='*70}")
    
    try:
        # 1. Extraer superficie
        logger.info("1️⃣  Extrayendo superficie...")
        positions, n_vacancies = extract_surface_atoms(dump_path)
        
        if len(positions) == 0:
            logger.error("   ❌ No se encontraron átomos de superficie")
            return None
        
        # 2. Normalizar (centra + PCA)
        logger.info("2️⃣  Normalizando posiciones (centrado + PCA)...")
        normalized_pos, box_size = normalize_positions(positions)
        
        # 3. Extraer features
        logger.info("3️⃣  Calculando features finales...")
        
        features = {}
        
        # Grid features (9 features)
        features.update(calc_grid_features(normalized_pos, box_size))
        
        # MOI principal (1 feature)
        features.update(calc_inertia_features(positions))
        
        # RDF features (2 features)
        features.update(calc_radial_features(positions))
        
        # Entropía espacial (1 feature)
        features.update(calc_entropy_feature(positions))
        
        # Bandwidth (1 feature)
        features.update(calc_bandwidth_feature(positions))
        
        # Target
        features['n_vacancies'] = float(n_vacancies)
        
        # Verificar que todas las features estén presentes
        missing = set(FINAL_FEATURES) - set(features.keys())
        if missing:
            logger.warning(f"   ⚠️  Features faltantes: {missing}")
        
        logger.info(f"   ✓ Total features: {len(features)}")
        logger.info(f"   ✓ Archivo procesado exitosamente")
        
        return features, len(positions)
        
    except Exception as e:
        logger.error(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== PROCESAMIENTO EN BATCH ====================

def process_directory(input_pattern, output_dir="features_final_optimized"):
    """
    Procesa directorio completo
    
    Args:
        input_pattern: patrón glob (ej: "dumps/*.dump")
        output_dir: directorio de salida
    """
    files = sorted(glob.glob(input_pattern))
    
    if not files:
        logger.error(f"❌ No se encontraron archivos: {input_pattern}")
        return None
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 EXTRACIÓN FINAL OPTIMIZADA")
    logger.info(f"{'='*70}")
    logger.info(f"Archivos encontrados: {len(files)}")
    logger.info(f"Features a extraer: {len(FINAL_FEATURES)}")
    logger.info(f"{'='*70}\n")
    
    all_rows = []
    successful = 0
    failed = 0
    failed_files = []
    
    for i, fp in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] {Path(fp).name}")
        
        result = extract_features_optimized(fp)
        
        if result is not None:
            features, n_atoms = result
            features['file'] = Path(fp).name
            all_rows.append(features)
            successful += 1
        else:
            failed += 1
            failed_files.append(Path(fp).name)
    
    # Resumen
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 RESUMEN DE PROCESAMIENTO")
    logger.info(f"{'='*70}")
    logger.info(f"✓ Exitosos: {successful}/{len(files)}")
    if failed > 0:
        logger.warning(f"⚠️  Fallidos: {failed}/{len(files)}")
        for fname in failed_files[:10]:
            logger.warning(f"   • {fname}")
        if len(failed_files) > 10:
            logger.warning(f"   ... y {len(failed_files)-10} más")
    
    if successful == 0:
        logger.error("❌ No se procesó ningún archivo exitosamente")
        return None
    
    # Crear DataFrame
    dataset = pd.DataFrame(all_rows).set_index('file').sort_index()
    
    # Asegurar orden de columnas
    feature_cols = [f for f in FINAL_FEATURES if f in dataset.columns]
    dataset = dataset[feature_cols]
    
    # Guardar
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_csv = Path(output_dir) / "dataset_final_features.csv"
    dataset.to_csv(output_csv)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✅ EXTRACCIÓN COMPLETADA")
    logger.info(f"{'='*70}")
    logger.info(f"📁 Dataset guardado: {output_csv}")
    logger.info(f"📊 Muestras: {len(dataset)}")
    logger.info(f"📈 Features: {len(dataset.columns)}")
    logger.info(f"\nFeatures extraídas:")
    for i, col in enumerate(dataset.columns, 1):
        logger.info(f"   {i:2d}. {col}")
    
    # Estadísticas
    logger.info(f"\n{'='*70}")
    logger.info(f"📈 ESTADÍSTICAS")
    logger.info(f"{'='*70}")
    logger.info(f"\nVacancias:")
    logger.info(f"   Min: {dataset['n_vacancies'].min():.0f}")
    logger.info(f"   Max: {dataset['n_vacancies'].max():.0f}")
    logger.info(f"   Media: {dataset['n_vacancies'].mean():.1f}")
    logger.info(f"   Std: {dataset['n_vacancies'].std():.1f}")
    
    logger.info(f"\nOccupancy:")
    logger.info(f"   Total Min: {dataset['occupancy_total'].min():.0f}")
    logger.info(f"   Total Max: {dataset['occupancy_total'].max():.0f}")
    logger.info(f"   Fraction Min: {dataset['occupancy_fraction'].min():.2f}")
    logger.info(f"   Fraction Max: {dataset['occupancy_fraction'].max():.2f}")
    
    logger.info(f"{'='*70}\n")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Extractor Final Optimizado - Solo 20 features finales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python extractor_final_optimized.py \\
      -i "datos/dumps/*.dump" \\
      -o features_output

Features extraídas (20):
  - occupancy_* (6 features)
  - occupancy_gradient_* (4 features)
  - occupancy_surface (1 feature)
  - grid_* (4 features)
  - moi_principal_3 (1 feature)
  - rdf_* (2 features)
  - entropy_spatial (1 feature)
  - ms_bandwidth (1 feature)

Validaciones implementadas:
  ✓ Centrado correcto del origen
  ✓ PCA optimizado con covariance_eigh
  ✓ Manejo de errores robusto
  ✓ Logs detallados
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Patrón de archivos dump (ej: "datos/dumps/*.dump")'
    )
    parser.add_argument(
        '-o', '--output',
        default='features_final_optimized',
        help='Directorio de salida'
    )
    
    args = parser.parse_args()
    
    try:
        dataset = process_directory(args.input, args.output)
        
        if dataset is not None:
            logger.info("✅ ÉXITO: Dataset generado correctamente")
        else:
            logger.error("❌ FALLO: No se generó dataset")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

    """
    python opentopologyc_extractor.py \
    -i "databases/db_integrate/*" \
    -o features_opentopologyc
    
    """