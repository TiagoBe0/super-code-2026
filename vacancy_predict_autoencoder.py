#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VacancyPredict ML - SOLO AUTOENCODER (Versión Simplificada)
============================================================
SOLO modo autoencoder: 19 features → 3 latentes → predicción
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from scipy.spatial import ConvexHull
from scipy.stats import kurtosis, entropy
from scipy.ndimage import label, gaussian_filter
from sklearn.cluster import estimate_bandwidth
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Constantes
A0 = 3.532
GRID_SIZE = (10, 10, 10)
BOX_SIZE_MAX = 10.0
GAUSSIAN_SIGMA = 0.6

# ============================================================================
# AUTOENCODER
# ============================================================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim=19):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z
    
    def encode_only(self, x):
        return self.encoder(x)


# ============================================================================
# EXTRACCIÓN DE FEATURES (19 FEATURES - IGUAL AL CSV)
# ============================================================================

def parse_dump_file(dump_content: str) -> np.ndarray:
    """Lee posiciones atómicas del dump"""
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
    
    return np.array(positions)


def normalize_positions_no_pca(positions):
    """Normaliza posiciones sin PCA"""
    if len(positions) < 2:
        centered = positions - positions.mean(axis=0)
        normalized = centered / A0
        return normalized, 2.0
    
    centered = positions - positions.mean(axis=0)
    normalized = centered / A0
    
    extent = normalized.max(axis=0) - normalized.min(axis=0)
    max_extent = extent.max()
    box_size = max(1.5, min(max_extent * 1.2, BOX_SIZE_MAX))
    
    return normalized, box_size


def build_voxel_grid_adaptive(positions, box_size, sigma=GAUSSIAN_SIGMA):
    """Voxelización con densidad Gaussiana"""
    N, M, L = GRID_SIZE
    grid_density = np.zeros((N, M, L), dtype=np.float32)
    
    half_box = box_size / 2.0
    cell_size = box_size / np.array(GRID_SIZE)
    
    for pos in positions:
        grid_pos = (pos + half_box) / cell_size
        
        range_sigma = 2.5 * sigma
        i_min = max(0, int(np.floor(grid_pos[0] - range_sigma)))
        i_max = min(N, int(np.ceil(grid_pos[0] + range_sigma)) + 1)
        j_min = max(0, int(np.floor(grid_pos[1] - range_sigma)))
        j_max = min(M, int(np.ceil(grid_pos[1] + range_sigma)) + 1)
        k_min = max(0, int(np.floor(grid_pos[2] - range_sigma)))
        k_max = min(L, int(np.ceil(grid_pos[2] + range_sigma)) + 1)
        
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                for k in range(k_min, k_max):
                    voxel_pos = np.array([i, j, k])
                    dist_sq = np.sum((voxel_pos - grid_pos)**2)
                    gaussian_contrib = np.exp(-dist_sq / (2 * sigma**2))
                    grid_density[i, j, k] += gaussian_contrib
    
    if grid_density.max() > 0:
        grid_density = grid_density / grid_density.max()
    
    grid_density = gaussian_filter(grid_density, sigma=0.7)
    
    if grid_density.max() > 0:
        grid_density = grid_density / grid_density.max()
    
    return grid_density


def calc_grid_features_19(grid):
    """
    ✅ EXACTAMENTE 10 features del grid (matching CSV)
    """
    N, M, L = grid.shape
    occupancy_total = float((grid > 0.01).sum())
    features = {}
    
    # 1. occupancy_total
    features['occupancy_total'] = occupancy_total
    
    # 2. occupancy_fraction
    features['occupancy_fraction'] = float((grid > 0).mean())
    
    # 3-5. occupancy_x/y/z_mean
    for axis, name in enumerate(['x', 'y', 'z']):
        slices = (grid > 0.01).sum(axis=axis)
        features[f'occupancy_{name}_mean'] = float(slices.mean())
    
    # 6-9. Gradientes
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
    
    # 10. occupancy_surface
    if occupancy_total > 0:
        surface = (np.abs(np.diff(grid, axis=0)).sum() + 
                  np.abs(np.diff(grid, axis=1)).sum() + 
                  np.abs(np.diff(grid, axis=2)).sum())
        features['occupancy_surface'] = float(surface)
    else:
        features['occupancy_surface'] = 0.0
    
    # 11. grid_entropy
    if occupancy_total > 0:
        prob = grid.flatten()
        prob = prob[prob > 0] / occupancy_total
        grid_entropy = -np.sum(prob * np.log(prob + 1e-10))
        features['grid_entropy'] = float(grid_entropy)
    else:
        features['grid_entropy'] = 0.0
    
    # 12-14. grid_moi (momentos de inercia del grid)
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
    
    return features  # 14 features


def calc_moi_principal_3(positions):
    """15. moi_principal_3 (solo el tercer momento)"""
    if len(positions) < 3:
        return {'moi_principal_3': np.nan}
    
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
    
    return {'moi_principal_3': float(eigenvalues[2])}


def calc_radial_features(positions):
    """16-17. rdf_mean y rdf_kurtosis"""
    if len(positions) < 2:
        return {'rdf_mean': np.nan, 'rdf_kurtosis': np.nan}
    
    com = positions.mean(axis=0)
    distances = np.linalg.norm(positions - com, axis=1)
    
    return {
        'rdf_mean': float(distances.mean()),
        'rdf_kurtosis': float(kurtosis(distances))
    }


def calc_entropy_spatial(positions):
    """18. entropy_spatial"""
    if len(positions) < 2:
        return {'entropy_spatial': np.nan}
    
    H, _ = np.histogramdd(positions, bins=10)
    H_flat = H.flatten()
    H_norm = H_flat[H_flat > 0] / H_flat.sum()
    
    return {'entropy_spatial': float(entropy(H_norm))}


def calc_bandwidth(positions):
    """19. ms_bandwidth"""
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


def extract_19_features(dump_content: str) -> Dict:
    """
    ✅ Extrae EXACTAMENTE 19 features (matching dataset_final_features.csv)
    
    Features (en orden):
    1-14: Grid features
    15: moi_principal_3
    16-17: rdf (mean, kurtosis)
    18: entropy_spatial
    19: ms_bandwidth
    """
    positions = parse_dump_file(dump_content)
    normalized_pos, box_size = normalize_positions_no_pca(positions)
    grid = build_voxel_grid_adaptive(normalized_pos, box_size)
    
    features = {}
    
    # 1-14: Grid features
    features.update(calc_grid_features_19(grid))
    
    # 15: moi_principal_3
    features.update(calc_moi_principal_3(positions))
    
    # 16-17: rdf
    features.update(calc_radial_features(positions))
    
    # 18: entropy_spatial
    features.update(calc_entropy_spatial(positions))
    
    # 19: ms_bandwidth
    features.update(calc_bandwidth(positions))
    
    return features


# ============================================================================
# PREDICCIÓN CON AUTOENCODER
# ============================================================================

def predict_with_autoencoder(dump_content: str, model, autoencoder, scaler):
    """
    Predicción: 19 features → scaler → encoder (3 latentes) → modelo
    """
    # Paso 1: Extraer 19 features
    features_dict = extract_19_features(dump_content)
    
    # Ordenar features en el orden correcto del CSV
    feature_names = [
        'occupancy_total', 'occupancy_fraction',
        'occupancy_x_mean', 'occupancy_y_mean', 'occupancy_z_mean',
        'occupancy_gradient_x', 'occupancy_gradient_y', 'occupancy_gradient_z',
        'occupancy_gradient_total', 'occupancy_surface',
        'grid_entropy', 'grid_moi_1', 'grid_moi_2', 'grid_moi_3',
        'moi_principal_3', 'rdf_mean', 'rdf_kurtosis',
        'entropy_spatial', 'ms_bandwidth'
    ]
    
    X_original = np.array([features_dict[f] for f in feature_names]).reshape(1, -1)
    
    # Paso 2: Normalizar
    X_scaled = scaler.transform(X_original)
    
    # Paso 3: Comprimir a 3 features latentes
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        X_latent = autoencoder.encode_only(X_tensor).numpy()
    
    # Paso 4: Predecir
    prediction = float(model.predict(X_latent)[0])
    
    return prediction, 19


def process_batch_files(uploaded_files, model, autoencoder, scaler):
    """Procesa múltiples archivos"""
    results = []
    
    for file in uploaded_files:
        result = {
            'archivo': file.name,
            'status': 'error',
            'prediccion': None,
            'prediccion_redondeada': None,
            'error': None
        }
        
        try:
            dump_content = file.read().decode('utf-8')
            prediction, n_features = predict_with_autoencoder(
                dump_content, model, autoencoder, scaler
            )
            
            result['status'] = 'success'
            result['features_extraidos'] = n_features
            result['prediccion'] = prediction
            result['prediccion_redondeada'] = int(round(prediction))
        
        except Exception as e:
            result['error'] = str(e)
        
        results.append(result)
    
    df_results = pd.DataFrame(results)
    
    successful = len(df_results[df_results['status'] == 'success'])
    df_success = df_results[df_results['status'] == 'success']
    
    stats = {
        'total_files': int(len(uploaded_files)),
        'successful': int(successful),
        'failed': int(len(uploaded_files) - successful),
        'total_vacancias': int(df_success['prediccion_redondeada'].sum()) if successful > 0 else 0,
        'mean_prediction': float(df_success['prediccion_redondeada'].mean()) if successful > 0 else 0.0,
    }
    
    return df_results, stats


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="VacancyPredict ML - Autoencoder",
    page_icon="🧠",
    layout="wide"
)

st.markdown("<h1 style='text-align: center; color: #667eea;'>🧠 VacancyPredict ML</h1>", 
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predicción con Autoencoder (19 → 3 features latentes)</p>", 
            unsafe_allow_html=True)

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("## 📦 Cargar Archivos")
    
    # Modelo ML
    st.markdown("### 🤖 Modelo ML")
    model_file = st.file_uploader("Modelo .pkl (entrenado con 3 features)", type=['pkl'])
    
    if model_file:
        try:
            model = joblib.load(model_file)
            st.success("✓ Modelo cargado")
            if hasattr(model, 'n_features_in_'):
                st.info(f"Features: {model.n_features_in_}")
            st.session_state.model = model
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    st.divider()
    
    # Autoencoder
    st.markdown("### 🧠 Autoencoder")
    autoencoder_file = st.file_uploader("autoencoder.pth", type=['pth', 'pt'])
    
    if autoencoder_file:
        try:
            autoencoder = Autoencoder(input_dim=19)
            autoencoder.load_state_dict(torch.load(autoencoder_file, map_location='cpu'))
            autoencoder.eval()
            st.success("✓ Autoencoder cargado")
            st.session_state.autoencoder = autoencoder
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    st.divider()
    
    # Scaler
    st.markdown("### 📊 Scaler")
    scaler_file = st.file_uploader("scaler.pkl", type=['pkl'])
    
    if scaler_file:
        try:
            scaler = joblib.load(scaler_file)
            st.success("✓ Scaler cargado")
            st.info(f"Features: {scaler.n_features_in_}")
            st.session_state.scaler = scaler
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    st.divider()
    
    # Estado
    st.markdown("## ✅ Estado")
    ready = ('model' in st.session_state and 
             'autoencoder' in st.session_state and 
             'scaler' in st.session_state)
    
    if ready:
        st.success("🟢 Sistema listo")
    else:
        st.warning("⚠️ Carga todos los archivos")

# Main
st.markdown("## 📋 Predicción por Lotes")

if not ('model' in st.session_state and 
        'autoencoder' in st.session_state and 
        'scaler' in st.session_state):
    st.warning("⚠️ Carga modelo, autoencoder y scaler en la barra lateral")
else:
    st.info("🧠 Modo: 19 features → 3 latentes → predicción")
    
    uploaded_files = st.file_uploader(
        "Arrastra archivos .dump",
        type=['dump', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.markdown(f"**📁 Archivos: {len(uploaded_files)}**")
        
        if st.button("🚀 Procesar", type="primary"):
            with st.spinner("Procesando..."):
                df_results, stats = process_batch_files(
                    uploaded_files,
                    st.session_state.model,
                    st.session_state.autoencoder,
                    st.session_state.scaler
                )
            
            st.markdown("## ✅ Resultados")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total", stats['total_files'])
            col2.metric("Exitosos", stats['successful'])
            col3.metric("Promedio", f"{stats['mean_prediction']:.1f}")
            
            st.divider()
            st.dataframe(df_results, use_container_width=True)
            
            st.divider()
            
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar CSV",
                data=csv_data,
                file_name=f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

st.divider()
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.9em;'>
VacancyPredict ML | 🧠 Autoencoder | 19→3 features
</div>
""", unsafe_allow_html=True)
