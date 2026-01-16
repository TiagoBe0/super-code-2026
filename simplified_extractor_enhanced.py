#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXTRACTOR SIMPLIFICADO MEJORADO - Features avanzadas del grid 3D
Versión optimizada con PCA mejorado y features adicionales del voxel grid

ARREGLADO: Ahora guarda archivos .npy y CSV con nombres consistentes
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# OVITO
from ovito.io import import_file
from ovito.modifiers import ConstructSurfaceModifier, InvertSelectionModifier, DeleteSelectedModifier, ExpressionSelectionModifier

# Configuración
ATM_TOTAL = 16384
A0 = 3.532
GRID_SIZE = (10, 10, 10)
BOX_SIZE_MAX = 10.0


def extract_surface_atoms(dump_path):
    """Extrae átomos de superficie usando OVITO"""
    pipeline = import_file(dump_path)
    
    pipeline.modifiers.append(ConstructSurfaceModifier(
        radius=2.0,
        select_surface_particles=True,
        smoothing_level=12,
        compute_distances=True
        
    ))
    data_full = pipeline.compute()
    n_vacancies = ATM_TOTAL - data_full.particles.count
    
    #pipeline.modifiers.append(ExpressionSelectionModifier( expression="SurfaceDistance > 4"))
        
    pipeline.modifiers.append(InvertSelectionModifier())
    pipeline.modifiers.append(DeleteSelectedModifier())
    
    data_surface = pipeline.compute()
    positions = data_surface.particles['Position'][...]
    
    logger.info(f"  Superficie: {len(positions)} átomos | Vacancias: {n_vacancies}")
    
    return positions, n_vacancies


def normalize_positions(positions):
    """
    Normaliza y alinea posiciones usando PCA OPTIMIZADO
    
    Mejoras:
    - Usa svd_solver='covariance_eigh' para mejor rendimiento cuando n_samples >> n_features
    - Fallback automático a 'auto' si hay problemas
    """
    if len(positions) < 3:
        return positions / A0, 2.0
    
    centered = positions - positions.mean(axis=0)
    
    # MEJORADO: Usar covariance_eigh es óptimo para muchos átomos (n>>p) con pocas features (3)
    try:
        pca = PCA(n_components=3, svd_solver='covariance_eigh')
        aligned = pca.fit_transform(centered)
    except Exception as e:
        # Fallback al solver automático si hay problemas numéricos
        logger.warning(f"  Fallback a PCA 'auto': {e}")
        pca = PCA(n_components=3, svd_solver='auto')
        aligned = pca.fit_transform(centered)
    
    normalized = aligned / A0
    
    extent = normalized.max(axis=0) - normalized.min(axis=0)
    box_size = max(1.5, min(extent.max() * 1.5, BOX_SIZE_MAX))
    
    return normalized, box_size


def calc_grid_features(positions, box_size):
    """
    Calcula features AVANZADAS del grid 3D
    
    NUEVAS FEATURES (26 total):
    - 6 básicas (originales)
    - 4 gradientes de ocupación
    - 1 fragmentación (clusters)
    - 2 compacidad
    - 3 centro de masa del grid
    - 1 asimetría
    - 1 entropía espacial
    - 3 momentos de inercia del grid
    - 5 densidad por capas (z seleccionadas)
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
    
    # ========== NUEVAS FEATURES ==========
    
    # 1. GRADIENTES DE OCUPACIÓN (4 features)
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
    
    # 2. FRAGMENTACIÓN - Número de clusters separados (1 feature)
    try:
        labeled_grid, n_clusters = label(grid)
        features['n_fragments'] = int(n_clusters)
    except:
        features['n_fragments'] = 0
    
    # 3. COMPACIDAD DEL VOLUMEN (2 features)
    if occupancy_total > 0:
        surface = (np.abs(np.diff(grid, axis=0)).sum() + 
                  np.abs(np.diff(grid, axis=1)).sum() + 
                  np.abs(np.diff(grid, axis=2)).sum())
        features['occupancy_surface'] = float(surface)
        # Compacidad: volumen^(2/3) / superficie (esfera = 1.0)
        features['occupancy_compactness'] = float(
            occupancy_total**(2/3) / (surface + 1e-8)
        )
    else:
        features['occupancy_surface'] = 0.0
        features['occupancy_compactness'] = 0.0
    
    # 4. CENTRO DE MASA DEL GRID (3 features)
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
    
    # 5. ASIMETRÍA DEL GRID en X (1 feature)
    if occupancy_total > 0:
        x_profile = grid.sum(axis=(1, 2))
        x_indices = np.arange(N)
        x_mean = (x_indices * x_profile).sum() / occupancy_total
        x_centered = x_indices - x_mean
        skewness = (x_centered**3 * x_profile).sum() / (occupancy_total * (N/4)**3 + 1e-8)
        features['grid_skewness_x'] = float(skewness)
    else:
        features['grid_skewness_x'] = 0.0
    
    # 6. ENTROPÍA ESPACIAL DEL GRID (1 feature)
    if occupancy_total > 0:
        prob = grid.flatten()
        prob = prob[prob > 0] / occupancy_total
        grid_entropy = -np.sum(prob * np.log(prob + 1e-10))
        features['grid_entropy'] = float(grid_entropy)
    else:
        features['grid_entropy'] = 0.0
    
    # 7. MOMENTOS DE INERCIA DEL GRID (3 features)
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
    
    # 8. DENSIDAD POR CAPAS Z SELECCIONADAS (5 features)
    # Capas importantes: inicio (z=0,1), medio (z=4,5) y final (z=8,9)
    for z in [0, 1, 4, 5, 8]:
        if z < L:
            features[f'occupancy_layer_z{z}'] = float(grid[:, :, z].sum())
        else:
            features[f'occupancy_layer_z{z}'] = 0.0
    
    return features


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


def calc_entropy_feature(positions):
    """Calcula entropía espacial"""
    if len(positions) < 2:
        return {'entropy_spatial': np.nan}
    
    H, _ = np.histogramdd(positions, bins=10)
    H_flat = H.flatten()
    H_norm = H_flat[H_flat > 0] / H_flat.sum()
    
    return {'entropy_spatial': float(entropy(H_norm))}


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


def extract_top_features(dump_path):
    """
    Extrae features mejoradas (ahora ~35 features en lugar de 15)
    
    Features extraídas:
    - 1 vacancia (target)
    - 26 del grid 3D (básicas + avanzadas)
    - 2 ConvexHull
    - 3 momentos de inercia
    - 2 radiales
    - 1 entropía espacial
    - 1 bandwidth
    
    ✅ ARREGLADO: Guarda grids y CSV con nombres consistentes (solo stem)
    """
    logger.info(f"\n🔨 Procesando: {Path(dump_path).name}")
    
    positions, n_vacancies = extract_surface_atoms(dump_path)
    normalized_pos, box_size = normalize_positions(positions)
    
    features = {'n_vacancies': n_vacancies}
    
    grid_features = calc_grid_features(normalized_pos, box_size)
    features.update(grid_features)

    # ✅ ARREGLADO: Guardar el grid para la CNN con nombre consistente
    grid = np.zeros(GRID_SIZE, dtype=np.int8)
    half_box = box_size / 2.0
    cell_size = box_size / np.array(GRID_SIZE)
    for pos in normalized_pos:
        indices = np.floor((pos + half_box) / cell_size).astype(int)
        if np.all(indices >= 0) and np.all(indices < GRID_SIZE):
            grid[indices[0], indices[1], indices[2]] = 1

    grid_out = Path("grids")
    grid_out.mkdir(exist_ok=True)
    # ✅ IMPORTANTE: Usar SOLO el stem (sin extensión) para consistencia
    grid_filename = Path(dump_path).stem
    np.save(grid_out / f"{grid_filename}.npy", grid)

    features.update(calc_hull_features(positions))
    features.update(calc_inertia_features(positions))
    features.update(calc_radial_features(positions))
    features.update(calc_entropy_feature(positions))
    features.update(calc_bandwidth_feature(positions))
    
    # ✅ IMPORTANTE: Guardar SOLO el stem en el CSV para coincidir con los archivos .npy
    features['file'] = grid_filename
    
    logger.info(f"  ✅ {len(features)-2} features extraídas (Grid: 26, Hull: 2, MOI: 3, otros: 4)")
    logger.info(f"     Grid guardado: grids/{grid_filename}.npy")
    
    return features


def process_directory(input_pattern, output_dir="simplified_features"):
    """Procesa múltiples archivos y genera dataset"""
    files = sorted(glob.glob(input_pattern))
    
    if not files:
        logger.error(f"No se encontraron archivos: {input_pattern}")
        return
    
    logger.info(f"🎯 Procesando {len(files)} archivos...")
    logger.info(f"📦 Extrayendo ~35 features mejoradas (Grid 3D optimizado)\n")
    
    all_rows = []
    
    for i, fp in enumerate(files, 1):
        try:
            features = extract_top_features(fp)
            all_rows.append(features)
            logger.info(f"  Progreso: {i}/{len(files)}\n")
        except Exception as e:
            logger.error(f"❌ Error en {fp}: {e}")
            continue
    
    if not all_rows:
        logger.error("No se generaron features")
        return
    
    dataset = pd.DataFrame(all_rows).set_index('file').sort_index()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_csv = Path(output_dir) / "dataset_enhanced_features.csv"
    dataset.to_csv(output_csv)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✅ DATASET MEJORADO GENERADO")
    logger.info(f"{'='*70}")
    logger.info(f"Archivo: {output_csv}")
    logger.info(f"Muestras: {len(dataset)}")
    logger.info(f"Features: {len(dataset.columns)}")
    logger.info(f"\nMejoras implementadas:")
    logger.info(f"  • PCA optimizado con covariance_eigh")
    logger.info(f"  • Grid 3D: +20 features (gradientes, fragmentación, compacidad)")
    logger.info(f"  • Total features: {len(dataset.columns)} (antes: 15)")
    logger.info(f"  • ✅ ARREGLADO: Nombres consistentes en grids y CSV")
    logger.info(f"\nVacancias - Min: {dataset['n_vacancies'].min():.0f} | "
                f"Max: {dataset['n_vacancies'].max():.0f} | "
                f"Media: {dataset['n_vacancies'].mean():.1f}")
    logger.info(f"{'='*70}")


def exportar_labelscsv_to_cnn():
    """Genera labels.csv con nombres consistentes"""
    df = pd.read_csv("simplified_features/dataset_enhanced_features.csv")
    df_labels = df.reset_index()[["file", "n_vacancies"]]
    df_labels.to_csv("labels.csv", index=False)
    print("✅ labels.csv generado correctamente")


def main():
    parser = argparse.ArgumentParser(
        description="Extractor Simplificado MEJORADO - Features avanzadas del Grid 3D"
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Patrón de archivos dump (ej: "dumps/*.dump")')
    parser.add_argument('-o', '--output', default='simplified_features',
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    process_directory(args.input, args.output)
    exportar_labelscsv_to_cnn()


if __name__ == "__main__":
    main()


    """
    
    python simplified_extractor_enhanced.py \
    -i "databases/db_integrate/dump.*" \
    -o simplified_features

    python simplified_extractor_enhanced.py \
    -i "databases/db_nov_test/*" \
    -o simplified_features

        python simplified_extractor_enhanced.py \
    -i "databases/db_cnn/*" \
    -o simplified_features
    """