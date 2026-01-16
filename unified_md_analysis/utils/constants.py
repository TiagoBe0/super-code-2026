"""
Constantes globales para análisis de estructuras cristalinas FCC/BCC
"""

# Parámetros cristalinos
ATM_TOTAL = 16384  # Número total de átomos esperado en simulación completa
A0 = 3.532  # Parámetro de red en Angstroms (típico para FCC como Cu)

# Parámetros del grid 3D de ocupación
GRID_SIZE = (10, 10, 10)  # Tamaño del grid para análisis de ocupación espacial
BOX_SIZE_MAX = 10.0  # Tamaño máximo de caja normalizada

# Parámetros de Alpha Shape
DEFAULT_PROBE_RADIUS = 2.0  # Radio de sonda típico para FCC/BCC (Angstroms)
GHOST_LAYER_THICKNESS = 1.5  # Espesor de capas fantasma para PBC (múltiplo de A0)

# Parámetros de clustering
DEFAULT_MIN_CLUSTER_SIZE = 10  # Tamaño mínimo de cluster en HDBSCAN
DEFAULT_MIN_SAMPLES = 5  # Mínimo de muestras para HDBSCAN

# Machine Learning
N_ESTIMATORS = 100  # Número de árboles en Random Forest
RANDOM_STATE = 42  # Semilla para reproducibilidad
TEST_SIZE = 0.2  # Proporción de datos para testing

# Features
N_GRID_FEATURES = 26  # Número de features extraídas del grid 3D
N_TOTAL_FEATURES = 35  # Número total de features extraídas (26+2+3+2+1+1)

# Lista ordenada de features (CRÍTICO: mantener este orden)
FEATURE_ORDER = [
    # Grid features (26)
    'occupancy_total', 'occupancy_fraction',
    'occupancy_x_mean', 'occupancy_y_mean', 'occupancy_z_mean',
    'occupancy_spread_k',
    'occupancy_gradient_x', 'occupancy_gradient_y', 'occupancy_gradient_z',
    'occupancy_gradient_total',
    'n_fragments',
    'occupancy_surface', 'occupancy_compactness',
    'grid_com_x', 'grid_com_y', 'grid_com_z',
    'grid_skewness_x',
    'grid_entropy',
    'grid_moi_1', 'grid_moi_2', 'grid_moi_3',
    'occupancy_layer_z0', 'occupancy_layer_z1', 'occupancy_layer_z4',
    'occupancy_layer_z5', 'occupancy_layer_z8',
    # Hull features (2)
    'hull_volume', 'hull_area',
    # Inertia features (3)
    'moi_principal_1', 'moi_principal_2', 'moi_principal_3',
    # Radial features (2)
    'rdf_mean', 'rdf_kurtosis',
    # Entropy features (1)
    'entropy_spatial',
    # Bandwidth features (1)
    'ms_bandwidth'
]

# Validación
assert len(FEATURE_ORDER) == N_TOTAL_FEATURES, \
    f"Número de features en FEATURE_ORDER ({len(FEATURE_ORDER)}) no coincide con N_TOTAL_FEATURES ({N_TOTAL_FEATURES})"
