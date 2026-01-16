#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VacancyPredict ML - COMPATIBLE con simplified_extractor_geometry_preserved.py
Versión mejorada para predicciones con geometría preservada

CAMBIOS PRINCIPALES:
====================
✅ Eliminado PCA rotation (preserva geometría)
✅ Soporta grids con densidad Gaussiana (float32)
✅ Soporta grids binarios (int8)
✅ Compatible con nuevo pipeline de extracción
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from scipy.spatial import ConvexHull
from scipy.stats import kurtosis, entropy
from scipy.ndimage import label, gaussian_filter
from sklearn.cluster import estimate_bandwidth
from typing import List, Dict, Tuple
import zipfile
import io
import warnings
warnings.filterwarnings('ignore')

# Constantes (IGUALES al extractor mejorado)
A0 = 3.532
GRID_SIZE = (10, 10, 10)
BOX_SIZE_MAX = 10.0
GAUSSIAN_SIGMA = 0.6

# ============================================================================
# FUNCIONES DE EXTRACCIÓN DE FEATURES (SINCRONIZADAS CON simplified_extractor_geometry_preserved.py)
# ============================================================================

def parse_dump_file(dump_content: str) -> np.ndarray:
    """
    Lee posiciones atómicas del dump
    (El dump ya viene con superficie extraída por otros scripts)
    """
    lines = dump_content.strip().split('\n')
    
    i = 0
    n_atoms = None
    atoms_start = None
    
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('ITEM: NUMBER OF ATOMS'):
            i += 1
            n_atoms = int(lines[i].strip())
        elif line.startswith('ITEM: ATOMS'):
            atoms_start = i + 1
            break
        i += 1
    
    if atoms_start is None or n_atoms is None:
        raise ValueError("Formato dump inválido")
    
    positions = []
    for i in range(atoms_start, atoms_start + n_atoms):
        if i < len(lines):
            parts = lines[i].strip().split()
            if len(parts) >= 5:
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                positions.append([x, y, z])
    
    positions = np.array(positions)
    
    if len(positions) == 0:
        raise ValueError("No se encontraron posiciones atómicas")
    
    return positions


def normalize_positions_no_pca(positions):
    """
    ✅ VERSIÓN MEJORADA: Sin PCA rotation (preserva geometría)
    
    ANTES (problemas):
    - Usaba PCA que rotaba arbitrariamente
    - Box size inflado
    - Geometría inconsistente
    
    AHORA (corregido):
    - Solo centrado, SIN rotación
    - Box size adaptativo
    - Geometría 100% preservada
    """
    if len(positions) < 2:
        centered = positions - positions.mean(axis=0)
        normalized = centered / A0
        return normalized, 2.0
    
    # Paso 1: Centrar en el centro de masa
    centered = positions - positions.mean(axis=0)
    
    # Paso 2: Normalizar por parámetro de red (SIN PCA)
    normalized = centered / A0
    
    # Paso 3: Calcular box size adaptativo
    extent = normalized.max(axis=0) - normalized.min(axis=0)
    max_extent = extent.max()
    
    # Box con margen pequeño
    box_size = max(1.5, min(max_extent * 1.2, BOX_SIZE_MAX))
    
    return normalized, box_size


def build_voxel_grid_adaptive(positions, box_size, sigma=GAUSSIAN_SIGMA):
    """
    ✅ NUEVA: Voxelización con densidad Gaussiana
    
    VENTAJAS:
    - Densidad continua [0, 1] (mejor para CNN)
    - Preserva información de proximidad
    - Suavidad mejora generalización
    - Compatible con grids antiguos
    """
    N, M, L = GRID_SIZE
    grid_density = np.zeros((N, M, L), dtype=np.float32)
    
    half_box = box_size / 2.0
    cell_size = box_size / np.array(GRID_SIZE)
    
    # Para cada átomo, calcular contribución Gaussiana
    for pos in positions:
        grid_pos = (pos + half_box) / cell_size
        
        # Rango de voxels afectados
        range_sigma = 2.5 * sigma
        i_min = max(0, int(np.floor(grid_pos[0] - range_sigma)))
        i_max = min(N, int(np.ceil(grid_pos[0] + range_sigma)) + 1)
        j_min = max(0, int(np.floor(grid_pos[1] - range_sigma)))
        j_max = min(M, int(np.ceil(grid_pos[1] + range_sigma)) + 1)
        k_min = max(0, int(np.floor(grid_pos[2] - range_sigma)))
        k_max = min(L, int(np.ceil(grid_pos[2] + range_sigma)) + 1)
        
        # Acumular contribución Gaussiana
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                for k in range(k_min, k_max):
                    voxel_pos = np.array([i, j, k])
                    dist_sq = np.sum((voxel_pos - grid_pos)**2)
                    gaussian_contrib = np.exp(-dist_sq / (2 * sigma**2))
                    grid_density[i, j, k] += gaussian_contrib
    
    # Normalizar
    if grid_density.max() > 0:
        grid_density = grid_density / grid_density.max()
    
    # Suavizado adicional
    grid_density = gaussian_filter(grid_density, sigma=0.7)
    
    # Renormalizar
    if grid_density.max() > 0:
        grid_density = grid_density / grid_density.max()
    
    return grid_density


def calc_grid_features(grid):
    """
    ✅ ACTUALIZADO: Calcula features desde grid directamente
    
    CAMBIOS:
    - Recibe grid ya voxelizado (no posiciones)
    - Compatible con float32 y int8
    - Threshold en 0.01 para grids continuos
    """
    N, M, L = grid.shape
    
    # Threshold para considerar ocupado
    occupancy_total = float((grid > 0.01).sum())
    features = {}
    
    # ========== FEATURES BÁSICAS ==========
    features['occupancy_total'] = occupancy_total
    features['occupancy_fraction'] = float((grid > 0).mean())
    
    for axis, name in enumerate(['x', 'y', 'z']):
        slices = (grid > 0.01).sum(axis=axis)
        features[f'occupancy_{name}_mean'] = float(slices.mean())
    
    k_indices = np.arange(L)
    slices_z = (grid > 0.01).sum(axis=(0, 1))
    if occupancy_total > 0:
        com_k = (k_indices * slices_z).sum() / occupancy_total
        features['occupancy_spread_k'] = float(
            np.sqrt(((k_indices - com_k)**2 * slices_z).sum() / occupancy_total)
        )
    else:
        features['occupancy_spread_k'] = 0.0
    
    # ========== GRADIENTES ==========
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
    
    # ========== FRAGMENTACIÓN ==========
    try:
        labeled_grid, n_clusters = label(grid > 0.01)
        features['n_fragments'] = int(n_clusters)
    except:
        features['n_fragments'] = 0
    
    # ========== COMPACIDAD ==========
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
    
    # ========== CENTRO DE MASA ==========
    if occupancy_total > 0:
        indices = np.argwhere(grid > 0.01)
        com = indices.mean(axis=0)
        features['grid_com_x'] = float(com[0] / N)
        features['grid_com_y'] = float(com[1] / M)
        features['grid_com_z'] = float(com[2] / L)
    else:
        features['grid_com_x'] = 0.5
        features['grid_com_y'] = 0.5
        features['grid_com_z'] = 0.5
    
    # ========== ASIMETRÍA ==========
    if occupancy_total > 0:
        x_profile = (grid > 0.01).sum(axis=(1, 2))
        x_indices = np.arange(N)
        x_mean = (x_indices * x_profile).sum() / occupancy_total
        x_centered = x_indices - x_mean
        skewness = (x_centered**3 * x_profile).sum() / (occupancy_total * (N/4)**3 + 1e-8)
        features['grid_skewness_x'] = float(skewness)
    else:
        features['grid_skewness_x'] = 0.0
    
    # ========== ENTROPÍA ==========
    if occupancy_total > 0:
        prob = (grid > 0.01).flatten()
        prob = prob[prob > 0] / occupancy_total
        grid_entropy = -np.sum(prob * np.log(prob + 1e-10))
        features['grid_entropy'] = float(grid_entropy)
    else:
        features['grid_entropy'] = 0.0
    
    # ========== MOMENTOS DE INERCIA ==========
    if occupancy_total > 0:
        try:
            coords = np.argwhere(grid > 0.01)
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
    
    # ========== DENSIDAD POR CAPAS ==========
    for z in [0, 1, 4, 5, 8]:
        if z < L:
            features[f'occupancy_layer_z{z}'] = float((grid[:, :, z] > 0.01).sum())
        else:
            features[f'occupancy_layer_z{z}'] = 0.0
    
    return features


def calc_hull_features(positions):
    """Calcula features del ConvexHull"""
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
    """Calcula momentos de inercia principales"""
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
    """Calcula features de distribución radial"""
    if len(positions) < 2:
        return {'rdf_mean': np.nan, 'rdf_kurtosis': np.nan}
    
    com = positions.mean(axis=0)
    distances = np.linalg.norm(positions - com, axis=1)
    
    from scipy.stats import kurtosis
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


def extract_features_from_dump(dump_content: str) -> Dict:
    """
    ✅ VERSIÓN MEJORADA: Extrae features con geometría preservada
    
    Pipeline:
    1. Parsear dump
    2. Normalizar SIN PCA (preserva geometría)
    3. Voxelizar con densidad Gaussiana
    4. Calcular features
    """
    try:
        # Paso 1: Parsear
        positions = parse_dump_file(dump_content)
        
        # Paso 2: Normalizar SIN PCA (CAMBIO CLAVE)
        normalized_pos, box_size = normalize_positions_no_pca(positions)
        
        # Paso 3: Voxelizar con densidad Gaussiana (CAMBIO CLAVE)
        grid = build_voxel_grid_adaptive(normalized_pos, box_size)
        
        # Paso 4: Calcular features
        features = {}
        
        # Features del grid
        features.update(calc_grid_features(grid))
        
        # Features del ConvexHull
        features.update(calc_hull_features(positions))
        
        # Momentos de inercia
        features.update(calc_inertia_features(positions))
        
        # Features radiales
        features.update(calc_radial_features(positions))
        
        # Entropía
        features.update(calc_entropy_feature(positions))
        
        # Bandwidth
        features.update(calc_bandwidth_feature(positions))
        
        return features
    
    except Exception as e:
        raise ValueError(f"Error extrayendo features: {str(e)}")


def process_batch_files(uploaded_files, model) -> Tuple[pd.DataFrame, Dict]:
    """
    Procesa lote de archivos con extracción mejorada
    """
    results = []
    
    for uploaded_file in uploaded_files:
        result = {
            'archivo': uploaded_file.name,
            'status': 'error',
            'features_extraidos': 0,
            'prediccion': np.nan,
            'prediccion_redondeada': 0,
            'error': ''
        }
        
        try:
            # Leer archivo
            dump_content = uploaded_file.read().decode('utf-8')
            
            # Extraer features (VERSIÓN MEJORADA)
            features_dict = extract_features_from_dump(dump_content)
            
            # Preparar para modelo
            feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
            
            if feature_names is not None:
                X = np.array([features_dict.get(name, 0.0) for name in feature_names]).reshape(1, -1)
            else:
                X = np.array(list(features_dict.values())).reshape(1, -1)
            
            # Predicción
            prediction = float(model.predict(X)[0])
            
            result['status'] = 'success'
            result['features_extraidos'] = len(features_dict)
            result['prediccion'] = prediction
            result['prediccion_redondeada'] = int(round(prediction))
        
        except Exception as e:
            result['error'] = str(e)
        
        results.append(result)
    
    df_results = pd.DataFrame(results)
    successful = len(df_results[df_results['status'] == 'success'])
    failed = len(df_results[df_results['status'] == 'error'])
    
    df_success = df_results[df_results['status'] == 'success']
    
    stats = {
        'total_files': int(len(uploaded_files)),
        'successful': int(successful),
        'failed': int(failed),
        'total_vacancias': int(df_success['prediccion_redondeada'].sum()) if successful > 0 else 0,
        'mean_prediction': float(df_success['prediccion_redondeada'].mean()) if successful > 0 else 0.0,
        'std_prediction': float(df_success['prediccion_redondeada'].std()) if successful > 0 else 0.0,
        'min_prediction': int(df_success['prediccion_redondeada'].min()) if successful > 0 else 0,
        'max_prediction': int(df_success['prediccion_redondeada'].max()) if successful > 0 else 0,
    }
    
    return df_results, stats


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="VacancyPredict ML - Batch",
    page_icon="🔬",
    layout="wide"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center; color: #667eea;'>VacancyPredict ML</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>✅ Compatible con simplified_extractor_geometry_preserved.py</p>", 
                unsafe_allow_html=True)

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("# 📦 Cargar Modelo")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    available_models = sorted([f for f in models_dir.glob("*.pkl")])
    
    if available_models:
        selected_model = st.selectbox(
            "Selecciona modelo pre-entrenado",
            available_models,
            format_func=lambda x: x.name
        )
        
        if st.button("✅ Cargar Modelo", use_container_width=True):
            try:
                model = joblib.load(selected_model)
                
                n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else "?"
                
                st.success(f"✓ Modelo cargado")
                st.info(f"**Features esperados:** {n_features}")
                
                if hasattr(model, 'feature_names_in_'):
                    with st.expander("Ver nombres de features"):
                        for feat in model.feature_names_in_:
                            st.text(f"• {feat}")
                
                st.session_state.model = model
                st.session_state.model_name = selected_model.name
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.info("No hay modelos en 'models/'")
    
    st.divider()
    
    st.markdown("### O carga manualmente")
    model_file = st.file_uploader("Selecciona (.pkl)", type=['pkl'])
    
    if model_file:
        try:
            model = joblib.load(model_file)
            
            n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else "?"
            
            st.success("✓ Modelo cargado")
            st.info(f"**Features esperados:** {n_features}")
            
            if hasattr(model, 'feature_names_in_'):
                with st.expander("Ver nombres de features"):
                    for feat in model.feature_names_in_:
                        st.text(f"• {feat}")
            
            st.session_state.model = model
            st.session_state.model_name = model_file.name
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    st.divider()
    if 'model' in st.session_state:
        st.markdown("### 🎯 Modelo Activo")
        st.success(f"**{st.session_state.model_name}**")
    else:
        st.warning("⚠️ Carga un modelo primero")

# Main content
st.markdown("## 📋 Procesamiento de Lotes")

if 'model' not in st.session_state:
    st.warning("⚠️ Por favor carga un modelo en la barra lateral")
else:
    st.info("✅ Versión mejorada: Extrae features con geometría preservada (sin PCA)")
    
    st.markdown("### Selecciona múltiples archivos dump")
    uploaded_files = st.file_uploader(
        "Arrastra archivos .dump o .txt (superficie ya extraída)",
        type=['dump', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.markdown(f"**📁 Archivos cargados: {len(uploaded_files)}**")
        
        with st.expander("Ver archivos cargados", expanded=False):
            for i, f in enumerate(uploaded_files, 1):
                st.text(f"{i}. {f.name}")
        
        if st.button("🚀 Procesar Lote", use_container_width=True, type="primary"):
            st.markdown("---")
            
            with st.spinner("Extrayendo features (geometría preservada) y prediciendo..."):
                df_results, stats = process_batch_files(
                    uploaded_files,
                    st.session_state.model
                )
            
            # Resultados
            st.markdown("## ✅ Resultados del Procesamiento")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📊 Total procesado", stats['total_files'])
            col2.metric("✓ Exitosos", stats['successful'])
            col3.metric("✗ Errores", stats['failed'])
            col4.metric("📍 Promedio vacancias", f"{stats['mean_prediction']:.1f}")
            
            st.divider()
            
            st.markdown("### 📄 Tabla de Resultados")
            st.dataframe(df_results, use_container_width=True, height=400)
            
            st.divider()
            
            if stats['successful'] > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Estadísticas")
                    st.metric("Predicción Mínima", f"{stats['min_prediction']:.0f}")
                    st.metric("Predicción Máxima", f"{stats['max_prediction']:.0f}")
                
                with col2:
                    st.markdown("### 📊 Dispersión")
                    st.metric("Media", f"{stats['mean_prediction']:.2f}")
                    st.metric("Desv. Estándar", f"{stats['std_prediction']:.2f}")
            
            st.divider()
            
            st.markdown("### 💾 Descargar Resultados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar CSV",
                    data=csv_data,
                    file_name=f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    csv_data = df_results.to_csv(index=False)
                    zf.writestr('resultados.csv', csv_data)
                    json_data = json.dumps(stats, indent=2)
                    zf.writestr('resumen.json', json_data)
                    timestamp_data = f"Procesamiento: {datetime.now().isoformat()}\n"
                    zf.writestr('metadata.txt', timestamp_data)
                zip_buffer.seek(0)
                
                st.download_button(
                    label="📦 Descargar ZIP (CSV + JSON)",
                    data=zip_buffer.getvalue(),
                    file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            
            st.session_state.last_results = df_results
            st.session_state.last_stats = stats

st.divider()
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.9em;'>
VacancyPredict ML v4.0 | ✅ Geometría Preservada | Compatible con simplified_extractor_geometry_preserved.py
</div>
""", unsafe_allow_html=True)
