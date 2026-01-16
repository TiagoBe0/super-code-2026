"""
Preprocesamiento: Extracción de Features para Machine Learning
Análisis de geometría de nanoporos - Extrae 35 features desde posiciones atómicas
"""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import kurtosis, entropy
from scipy.ndimage import label
from sklearn.decomposition import PCA
from sklearn.cluster import estimate_bandwidth
from typing import Dict, Tuple

from ..utils.constants import A0, GRID_SIZE, BOX_SIZE_MAX, FEATURE_ORDER


def normalize_positions(positions: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Normaliza y alinea posiciones usando PCA optimizado

    - Centra posiciones en origen
    - Alinea con componentes principales (PCA)
    - Normaliza por parámetro de red A0
    - Calcula box_size apropiado

    Args:
        positions: array Nx3 de posiciones atómicas

    Returns:
        normalized: posiciones normalizadas y alineadas
        box_size: tamaño de caja normalizada
    """
    if len(positions) < 3:
        return positions / A0, 2.0

    centered = positions - positions.mean(axis=0)

    # MEJORADO: Usar covariance_eigh es óptimo para muchos átomos
    try:
        pca = PCA(n_components=3, svd_solver='covariance_eigh')
        aligned = pca.fit_transform(centered)
    except Exception:
        # Fallback al solver automático si hay problemas numéricos
        pca = PCA(n_components=3, svd_solver='auto')
        aligned = pca.fit_transform(centered)

    normalized = aligned / A0

    extent = normalized.max(axis=0) - normalized.min(axis=0)
    box_size = max(1.5, min(extent.max() * 1.5, BOX_SIZE_MAX))

    return normalized, box_size


def calc_grid_features(positions: np.ndarray, box_size: float) -> Dict[str, float]:
    """
    Calcula 26 features avanzadas del grid 3D de ocupación

    Features:
    - 6 básicas: ocupación total, fracción, medias por eje, spread
    - 4 gradientes: cambios de ocupación en x, y, z
    - 1 fragmentación: número de clusters separados
    - 2 compacidad: superficie y ratio volumen/superficie
    - 3 centro de masa: posición normalizada del centroide
    - 1 asimetría: skewness en eje x
    - 1 entropía: entropía espacial del grid
    - 3 momentos de inercia: valores propios del tensor de inercia
    - 5 densidad por capas: ocupación en capas z específicas

    Args:
        positions: array Nx3 de posiciones normalizadas
        box_size: tamaño de caja normalizada

    Returns:
        features: diccionario con 26 features
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
        # Compacidad: volumen^(2/3) / superficie (esfera = 1.0)
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

    # ========== DENSIDAD POR CAPAS (5) ==========
    # Capas importantes: inicio (z=0,1), medio (z=4,5) y final (z=8)
    for z in [0, 1, 4, 5, 8]:
        if z < L:
            features[f'occupancy_layer_z{z}'] = float(grid[:, :, z].sum())
        else:
            features[f'occupancy_layer_z{z}'] = 0.0

    return features


def calc_hull_features(positions: np.ndarray) -> Dict[str, float]:
    """
    Calcula 2 features del Convex Hull

    Args:
        positions: array Nx3 de posiciones (sin normalizar)

    Returns:
        features: hull_volume, hull_area
    """
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


def calc_inertia_features(positions: np.ndarray) -> Dict[str, float]:
    """
    Calcula 3 momentos de inercia principales

    Args:
        positions: array Nx3 de posiciones

    Returns:
        features: moi_principal_1, moi_principal_2, moi_principal_3
    """
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


def calc_radial_features(positions: np.ndarray) -> Dict[str, float]:
    """
    Calcula 2 features de distribución radial (RDF)

    Args:
        positions: array Nx3 de posiciones

    Returns:
        features: rdf_mean, rdf_kurtosis
    """
    if len(positions) < 2:
        return {'rdf_mean': np.nan, 'rdf_kurtosis': np.nan}

    com = positions.mean(axis=0)
    distances = np.linalg.norm(positions - com, axis=1)

    return {
        'rdf_mean': float(distances.mean()),
        'rdf_kurtosis': float(kurtosis(distances))
    }


def calc_entropy_feature(positions: np.ndarray) -> Dict[str, float]:
    """
    Calcula 1 feature de entropía espacial

    Args:
        positions: array Nx3 de posiciones

    Returns:
        features: entropy_spatial
    """
    if len(positions) < 2:
        return {'entropy_spatial': np.nan}

    H, _ = np.histogramdd(positions, bins=10)
    H_flat = H.flatten()
    H_norm = H_flat[H_flat > 0] / H_flat.sum()

    return {'entropy_spatial': float(entropy(H_norm))}


def calc_bandwidth_feature(positions: np.ndarray) -> Dict[str, float]:
    """
    Calcula 1 feature de bandwidth de clustering (MeanShift)

    Args:
        positions: array Nx3 de posiciones

    Returns:
        features: ms_bandwidth
    """
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


def extract_all_features(positions: np.ndarray) -> Dict[str, float]:
    """
    Extrae todas las 35 features de un conjunto de posiciones

    PROCESO:
    1. Normalizar posiciones (PCA + escalado por A0)
    2. Calcular 26 features del grid 3D
    3. Calcular 2 features de ConvexHull
    4. Calcular 3 momentos de inercia principales
    5. Calcular 2 features radiales (RDF)
    6. Calcular 1 entropía espacial
    7. Calcular 1 bandwidth de clustering

    Args:
        positions: array Nx3 de posiciones atómicas (Angstroms)

    Returns:
        features: diccionario con 35 features en orden FEATURE_ORDER
    """
    # 1. Normalizar posiciones
    normalized_pos, box_size = normalize_positions(positions)

    # 2. Calcular todas las features
    features = {}

    # Grid features (26)
    features.update(calc_grid_features(normalized_pos, box_size))

    # Hull features (2)
    features.update(calc_hull_features(positions))

    # Inertia features (3)
    features.update(calc_inertia_features(positions))

    # Radial features (2)
    features.update(calc_radial_features(positions))

    # Entropy features (1)
    features.update(calc_entropy_feature(positions))

    # Bandwidth features (1)
    features.update(calc_bandwidth_feature(positions))

    # Validar que tenemos todas las features
    if len(features) != len(FEATURE_ORDER):
        missing = set(FEATURE_ORDER) - set(features.keys())
        extra = set(features.keys()) - set(FEATURE_ORDER)
        raise ValueError(
            f"Número de features incorrecto. "
            f"Esperado: {len(FEATURE_ORDER)}, Obtenido: {len(features)}. "
            f"Faltantes: {missing}, Extras: {extra}"
        )

    return features


def features_to_array(features: Dict[str, float]) -> np.ndarray:
    """
    Convierte diccionario de features a array ordenado

    CRÍTICO: Usa el orden definido en FEATURE_ORDER para consistencia

    Args:
        features: diccionario con features

    Returns:
        array de features en orden correcto
    """
    return np.array([features[name] for name in FEATURE_ORDER])
