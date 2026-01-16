#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Completo de Extracción de Features para Predicción de Vacancias
Combina: OVITO + Grid 3D Mejorado + ConvexHull + Mean Shift + Energía
"""

import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
warnings.filterwarnings('ignore', category=RuntimeWarning)

import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import kurtosis, entropy
from scipy.ndimage import label
from sklearn.decomposition import PCA
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import NearestNeighbors
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# OVITO
from ovito.io import import_file
from ovito.modifiers import (
    ConstructSurfaceModifier, 
    InvertSelectionModifier, 
    DeleteSelectedModifier,
    ExpressionSelectionModifier
)

# ==================== CONFIGURACIóN ====================
ATM_TOTAL = 16384
A0 = 3.532
GRID_SIZE = (10, 10, 10)
BOX_SIZE_MAX = 10.0

# Features prohibidas (no usar para entrenamiento)
FORBIDDEN_FEATURES = ['n_vacancies', 'vacancies', 'n_atoms_surface']

# Energía (si está disponible)
ENERGY_MIN = -4.5
ENERGY_MAX = -4.3
ENERGY_BINS = 20


# ==================== OVITO PROCESSING ====================

def extract_surface_atoms_with_energy(dump_path, ovito_dir='ovito_processed'):
    """
    Extrae átomos de superficie usando OVITO y guarda archivo procesado
    Incluye información de energía si está disponible
    """
    pipeline = import_file(dump_path)
    
    # Superficie
    pipeline.modifiers.append(ConstructSurfaceModifier(
        radius=2.0,
        select_surface_particles=True,
        smoothing_level=8,compute_distances=True
        
    ))
    
    data_full = pipeline.compute()
    n_vacancies = ATM_TOTAL - data_full.particles.count
    
    # Invertir selección y borrar bulk
    #pipeline.modifiers.append(InvertSelectionModifier())
    pipeline.modifiers.append(ExpressionSelectionModifier(expression="SurfaceDistance>=4"))
    pipeline.modifiers.append(DeleteSelectedModifier())
    
    data_surface = pipeline.compute()
    positions = data_surface.particles['Position'][...]
    
    # Intentar extraer energía
    c_pe = None
    try:
        if 'c_pe' in data_surface.particles:
            c_pe = data_surface.particles['c_pe'][...]
        elif 'Energy' in data_surface.particles:
            c_pe = data_surface.particles['Energy'][...]
    except:
        pass
    
    # Guardar archivo procesado
    Path(ovito_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(ovito_dir) / Path(dump_path).name
    
    # Crear DataFrame para guardar
    df_surface = pd.DataFrame({
        'x': positions[:, 0],
        'y': positions[:, 1],
        'z': positions[:, 2]
    })
    if c_pe is not None:
        df_surface['c_pe'] = c_pe
    
    df_surface.to_csv(str(output_file) + '.csv', index=False)
    
    logger.info(f"  Superficie: {len(positions)} átomos | Vacancias: {n_vacancies}")
    if c_pe is not None:
        logger.info(f"  Energía: Min={c_pe.min():.3f}, Max={c_pe.max():.3f}")
    
    return positions, c_pe, n_vacancies


# ==================== NORMALIZACIÓN Y PCA ====================

def normalize_positions(positions):
    """
    Normaliza y alinea posiciones usando PCA OPTIMIZADO
    """
    if len(positions) < 3:
        return positions / A0, 2.0
    
    centered = positions - positions.mean(axis=0)
    
    try:
        pca = PCA(n_components=3, svd_solver='covariance_eigh')
        aligned = pca.fit_transform(centered)
    except Exception as e:
        logger.warning(f"  Fallback a PCA 'auto': {e}")
        pca = PCA(n_components=3, svd_solver='auto')
        aligned = pca.fit_transform(centered)
    
    normalized = aligned / A0
    
    extent = normalized.max(axis=0) - normalized.min(axis=0)
    box_size = max(1.5, min(extent.max() * 1.5, BOX_SIZE_MAX))
    
    return normalized, box_size


# ==================== GRID 3D MEJORADO ====================

def calc_grid_features_enhanced(positions, box_size):
    """
    Calcula features AVANZADAS del grid 3D (26 features)
    
    Incluye:
    - 6 básicas (ocupación total, fracciones, spread)
    - 4 gradientes de ocupación
    - 1 fragmentación (clusters)
    - 2 compacidad
    - 3 centro de masa del grid
    - 1 asimetría
    - 1 entropía espacial
    - 3 momentos de inercia del grid
    - 5 densidad por capas
    """
    N, M, L = GRID_SIZE
    grid = np.zeros((N, M, L), dtype=np.int8)
    
    half_box = box_size / 2.0
    cell_size = box_size / np.array(GRID_SIZE)
    
    # Llenar grid
    for pos in positions:
        indices = np.floor((pos + half_box) / cell_size).astype(int)
        if np.all(indices >= 0) and np.all(indices < GRID_SIZE):
            grid[indices[0], indices[1], indices[2]] = 1
    
    occupancy_total = grid.sum()
    features = {}
    
    # ========== FEATURES BÁSICAS (6) ==========
    features['occupancy_total'] = float(occupancy_total)
    features['occupancy_fraction'] = float(grid.mean())
    
    for axis, name in enumerate(['x', 'y', 'z']):
        slices = grid.sum(axis=axis)
        features[f'occupancy_{name}_mean'] = float(slices.mean())
    
    k_indices = np.arange(L)
    slices_z = grid.sum(axis=(0, 1))
    if occupancy_total > 0:
        com_k = (k_indices * slices_z).sum() / occupancy_total
        features['occupancy_spread_k'] = float(
            np.sqrt(((k_indices - com_k)**2 * slices_z).sum() / occupancy_total)
        )
    else:
        features['occupancy_spread_k'] = 0.0
    
    # ========== GRADIENTES DE OCUPACIÓN (4) ==========
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
    
    # ========== FRAGMENTACIÓN (1) ==========
    try:
        labeled_grid, n_clusters = label(grid)
        features['n_fragments'] = int(n_clusters)
    except:
        features['n_fragments'] = 0
    
    # ========== COMPACIDAD (2) ==========
    if occupancy_total > 0:
        surface = (np.abs(np.diff(grid, axis=0)).sum() + 
                  np.abs(np.diff(grid, axis=1)).sum() + 
                  np.abs(np.diff(grid, axis=2)).sum())
        features['occupancy_surface'] = float(surface)
        features['occupancy_compactness'] = float(
            occupancy_total**(2/3) / (surface + 1e-8)
        )
    else:
        features['occupancy_surface'] = 0.0
        features['occupancy_compactness'] = 0.0
    
    # ========== CENTRO DE MASA DEL GRID (3) ==========
    if occupancy_total > 0:
        indices = np.argwhere(grid == 1)
        com = indices.mean(axis=0)
        features['grid_com_x'] = float(com[0] / N)
        features['grid_com_y'] = float(com[1] / M)
        features['grid_com_z'] = float(com[2] / L)
    else:
        features['grid_com_x'] = 0.5
        features['grid_com_y'] = 0.5
        features['grid_com_z'] = 0.5
    
    # ========== ASIMETRÍA (1) ==========
    if occupancy_total > 0:
        x_profile = grid.sum(axis=(1, 2))
        x_indices = np.arange(N)
        x_mean = (x_indices * x_profile).sum() / occupancy_total
        x_centered = x_indices - x_mean
        skewness = (x_centered**3 * x_profile).sum() / (occupancy_total * (N/4)**3 + 1e-8)
        features['grid_skewness_x'] = float(skewness)
    else:
        features['grid_skewness_x'] = 0.0
    
    # ========== ENTROPÍA ESPACIAL (1) ==========
    if occupancy_total > 0:
        prob = grid.flatten()
        prob = prob[prob > 0] / occupancy_total
        grid_entropy = -np.sum(prob * np.log(prob + 1e-10))
        features['grid_entropy'] = float(grid_entropy)
    else:
        features['grid_entropy'] = 0.0
    
    # ========== MOMENTOS DE INERCIA DEL GRID (3) ==========
    if occupancy_total > 0:
        try:
            coords = np.argwhere(grid == 1)
            centered = coords - coords.mean(axis=0)
            
            Ixx = np.sum(centered[:, 1]**2 + centered[:, 2]**2)
            Iyy = np.sum(centered[:, 0]**2 + centered[:, 2]**2)
            Izz = np.sum(centered[:, 0]**2 + centered[:, 1]**2)
            Ixy = -np.sum(centered[:, 0] * centered[:, 1])
            Ixz = -np.sum(centered[:, 0] * centered[:, 2])
            Iyz = -np.sum(centered[:, 1] * centered[:, 2])
            
            I_tensor = np.array([[Ixx, Ixy, Ixz],
                                [Ixy, Iyy, Iyz],
                                [Ixz, Iyz, Izz]])
            
            eigenvalues = np.sort(np.linalg.eigvalsh(I_tensor))[::-1]
            features['grid_moi_1'] = float(eigenvalues[0])
            features['grid_moi_2'] = float(eigenvalues[1])
            features['grid_moi_3'] = float(eigenvalues[2])
        except:
            features['grid_moi_1'] = 0.0
            features['grid_moi_2'] = 0.0
            features['grid_moi_3'] = 0.0
    else:
        features['grid_moi_1'] = 0.0
        features['grid_moi_2'] = 0.0
        features['grid_moi_3'] = 0.0
    
    # ========== DENSIDAD POR CAPAS Z (5) ==========
    for z in [0, 1, 4, 5, 8]:
        if z < L:
            features[f'occupancy_layer_z{z}'] = float(grid[:, :, z].sum())
        else:
            features[f'occupancy_layer_z{z}'] = 0.0
    
    return features


# ==================== CONVEX HULL ====================

def calc_hull_features(positions):
    """Calcula 2 features del ConvexHull"""
    if len(positions) < 4:
        return {'hull_volume': np.nan, 'hull_area': np.nan}
    
    try:
        hull = ConvexHull(positions)
        return {
            'hull_volume': float(hull.volume),
            'hull_area': float(hull.area)
        }
    except:
        return {'hull_volume': np.nan, 'hull_area': np.nan}


# ==================== MOMENTOS DE INERCIA ====================

def calc_inertia_features(positions):
    """Calcula 3 momentos de inercia principales"""
    if len(positions) < 3:
        return {f'moi_principal_{i}': np.nan for i in [1, 2, 3]}
    
    centered = positions - positions.mean(axis=0)
    
    Ixx = np.sum(centered[:, 1]**2 + centered[:, 2]**2)
    Iyy = np.sum(centered[:, 0]**2 + centered[:, 2]**2)
    Izz = np.sum(centered[:, 0]**2 + centered[:, 1]**2)
    
    Ixy = -np.sum(centered[:, 0] * centered[:, 1])
    Ixz = -np.sum(centered[:, 0] * centered[:, 2])
    Iyz = -np.sum(centered[:, 1] * centered[:, 2])
    
    I_tensor = np.array([[Ixx, Ixy, Ixz],
                        [Ixy, Iyy, Iyz],
                        [Ixz, Iyz, Izz]])
    
    eigenvalues = np.sort(np.linalg.eigvalsh(I_tensor))[::-1]
    
    return {
        'moi_principal_1': float(eigenvalues[0]),
        'moi_principal_2': float(eigenvalues[1]),
        'moi_principal_3': float(eigenvalues[2])
    }


# ==================== RADIAL ====================

def calc_radial_features(positions):
    """Calcula 2 features de distribución radial"""
    if len(positions) < 2:
        return {'rdf_mean': np.nan, 'rdf_kurtosis': np.nan}
    
    com = positions.mean(axis=0)
    distances = np.linalg.norm(positions - com, axis=1)
    
    return {
        'rdf_mean': float(distances.mean()),
        'rdf_kurtosis': float(kurtosis(distances))
    }


# ==================== ENTROPÍA ESPACIAL ====================

def calc_entropy_feature(positions):
    """Calcula entropía espacial"""
    if len(positions) < 2:
        return {'entropy_spatial': np.nan}
    
    H, _ = np.histogramdd(positions, bins=10)
    H_flat = H.flatten()
    H_norm = H_flat[H_flat > 0] / H_flat.sum()
    
    return {'entropy_spatial': float(entropy(H_norm))}


# ==================== MEAN SHIFT BANDWIDTH ====================

def calc_bandwidth_feature(positions):
    """Calcula bandwidth de clustering"""
    if len(positions) < 10:
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
    except:
        return {'ms_bandwidth': np.nan}


# ==================== ENERGÍA ====================

def calc_energy_features(c_pe, energy_min, energy_max, n_bins):
    """
    Calcula histograma de energía (si disponible)
    """
    if c_pe is None or len(c_pe) == 0:
        return {f'energy_bin_{i}': np.nan for i in range(n_bins)}
    
    hist, _ = np.histogram(c_pe, bins=n_bins, range=(energy_min, energy_max))
    hist_norm = hist / (hist.sum() + 1e-10)
    
    return {f'energy_bin_{i}': float(hist_norm[i]) for i in range(n_bins)}


# ==================== PIPELINE COMPLETO ====================

def process_file_complete(dump_path, ovito_dir='ovito_processed'):
    """
    Procesa un archivo dump y extrae TODAS las features
    
    Total de features: ~90
    - 1 vacancia (target)
    - 1 n_atoms_surface
    - 26 grid 3D mejorado
    - 2 convex hull
    - 3 momentos de inercia
    - 2 radiales
    - 1 entropía espacial
    - 1 bandwidth
    - 50 energía (si disponible)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 Procesando: {Path(dump_path).name}")
    logger.info(f"{'='*70}")
    
    # 1. OVITO - Extraer superficie
    positions, c_pe, n_vacancies = extract_surface_atoms_with_energy(dump_path, ovito_dir)
    
    # 2. Normalizar posiciones
    normalized_pos, box_size = normalize_positions(positions)
    
    # 3. Construir features
    features = {
        'n_vacancies': n_vacancies,
        'n_atoms_surface': len(positions)
    }
    
    # Grid 3D MEJORADO (26 features)
    features.update(calc_grid_features_enhanced(normalized_pos, box_size))
    
    # ConvexHull (2 features)
    features.update(calc_hull_features(positions))
    
    # Momentos de inercia (3 features)
    features.update(calc_inertia_features(positions))
    
    # Radiales (2 features)
    features.update(calc_radial_features(positions))
    
    # Entropía espacial (1 feature)
    features.update(calc_entropy_feature(positions))
    
    # Mean Shift bandwidth (1 feature)
    features.update(calc_bandwidth_feature(positions))
    
    # Energía (50 features si disponible)
    features.update(calc_energy_features(c_pe, ENERGY_MIN, ENERGY_MAX, ENERGY_BINS))
    
    features['file'] = Path(dump_path).name
    
    n_features = len(features) - 2  # Sin contar file y n_vacancies
    logger.info(f"  ✅ {n_features} features extraídas")
    logger.info(f"     Grid3D: 26 | Hull: 2 | MOI: 3 | Radial: 2")
    logger.info(f"     Entropía: 1 | Bandwidth: 1 | Energía: 50")
    
    return features


# ==================== PROCESAMIENTO DE DIRECTORIO ====================

def process_directory(input_pattern, output_dir="features_complete", ovito_dir="ovito_processed"):
    """Procesa múltiples archivos y genera dataset completo"""
    files = sorted(glob.glob(input_pattern))
    
    if not files:
        logger.error(f"No se encontraron archivos: {input_pattern}")
        return
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 PIPELINE COMPLETO DE EXTRACCIÓN DE FEATURES")
    logger.info(f"{'='*70}")
    logger.info(f"📁 Procesando {len(files)} archivos...")
    logger.info(f"📊 Extrayendo ~90 features por archivo")
    logger.info(f"{'='*70}\n")
    
    all_rows = []
    
    for i, fp in enumerate(files, 1):
        try:
            features = process_file_complete(fp, ovito_dir)
            all_rows.append(features)
            logger.info(f"  Progreso: {i}/{len(files)}\n")
        except Exception as e:
            logger.error(f"❌ Error en {fp}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_rows:
        logger.error("No se generaron features")
        return
    
    # Crear dataset
    dataset = pd.DataFrame(all_rows).set_index('file').sort_index()
    
    # Guardar
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_csv = Path(output_dir) / "dataset_complete_features.csv"
    dataset.to_csv(output_csv)
    
    # Resumen
    logger.info(f"\n{'='*70}")
    logger.info(f"✅ DATASET COMPLETO GENERADO")
    logger.info(f"{'='*70}")
    logger.info(f"📁 Archivo: {output_csv}")
    logger.info(f"📊 Muestras: {len(dataset)}")
    logger.info(f"📈 Features: {len(dataset.columns)}")
    logger.info(f"\n📋 Distribución de features:")
    logger.info(f"   • Target: n_vacancies")
    logger.info(f"   • Superficie: n_atoms_surface")
    logger.info(f"   • Grid 3D Mejorado: 26 features")
    logger.info(f"   • ConvexHull: 2 features")
    logger.info(f"   • Momentos Inercia: 3 features")
    logger.info(f"   • Radial: 2 features")
    logger.info(f"   • Entropía: 1 feature")
    logger.info(f"   • Bandwidth: 1 feature")
    logger.info(f"   • Energía: 50 features")
    logger.info(f"\n📊 Estadísticas de Vacancias:")
    logger.info(f"   Min: {dataset['n_vacancies'].min():.0f} | "
                f"Max: {dataset['n_vacancies'].max():.0f} | "
                f"Media: {dataset['n_vacancies'].mean():.1f}")
    logger.info(f"{'='*70}\n")


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Completo - Extracción de Features para Predicción de Vacancias"
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Patrón de archivos dump (ej: "dumps/*.dump")')
    parser.add_argument('-o', '--output', default='features_complete',
                       help='Directorio de salida para features')
    parser.add_argument('--ovito-dir', default='ovito_processed',
                       help='Directorio para archivos procesados con OVITO')
    
    args = parser.parse_args()
    
    process_directory(args.input, args.output, args.ovito_dir)


if __name__ == "__main__":
    main()


"""
EJEMPLOS DE USO:

1. Procesar todos los dumps en un directorio:
   python vacancy_pipeline_combined.py -i "databases/db_integrate/dump.*" -o features_output

2. Con directorios personalizados:
   python vacancy_pipeline_combined.py \
       -i "dumps/*.dump" \
       -o features_complete \
       --ovito-dir ovito_processed

3. Procesar base de datos MULTIVOIDS:
   python vacancy_pipeline_combined.py \
       -i "databases/MULTIVOIDS/dump.*" \
       -o features_multivoids


       python simplified_extractor_enhanced_spirit.py \
       -i "databases/db_integrate/dump.*" \
       -o features_complete_spirit \
       --ovito-dir ovito_processed_spirit
"""