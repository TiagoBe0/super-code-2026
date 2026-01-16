"""
🧩 CLUSTERING AVANZADO - Análisis de Nanoporos
Interfaz visual para clustering multinodo con HDBSCAN, KMeans, MeanShift, Aglomerativo
CON EXPORTACIÓN DE CLUSTERS INDIVIDUALES EN ARCHIVOS .DUMP SEPARADOS
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import traceback
import shutil
from pathlib import Path
from io import StringIO, BytesIO
from typing import Dict, List, Tuple, Optional, Any
import json
import tempfile
import colorsys
import plotly.graph_objects as go
import plotly.express as px
import zipfile
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Intentar importar HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Clustering Avanzado",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado (idéntico a alpha_shape para consistencia)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧩 Clustering Avanzado para Nanoporos</div>', unsafe_allow_html=True)
st.markdown("**Análisis automático de estructuras con múltiples algoritmos de clustering**")
st.markdown("**📁 Exportación: Clusters individuales en archivos .dump separados**")
st.markdown("---")

# ==========================================
# PARSERS PARA LAMMPS DUMP
# ==========================================

@st.cache_data
def parse_lammps_dump(file_content: bytes) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Parser robusto para archivos LAMMPS dump
    
    Returns:
        header: diccionario con metadata (timestep, box_bounds, etc.)
        df: DataFrame con datos de átomos
    """
    try:
        lines = file_content.decode('utf-8').split('\n')
    except UnicodeDecodeError:
        lines = file_content.decode('latin-1').split('\n')
    
    header = {
        'timestep': 0,
        'n_atoms': 0,
        'box_bounds': [],
        'pbc': ['pp', 'pp', 'pp']
    }
    atom_lines = []
    columns = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line == "ITEM: TIMESTEP":
            header['timestep'] = int(lines[i+1].strip())
            i += 2
            
        elif line == "ITEM: NUMBER OF ATOMS":
            header['n_atoms'] = int(lines[i+1].strip())
            i += 2
            
        elif line.startswith("ITEM: BOX BOUNDS"):
            parts = line.split()
            if len(parts) > 3:
                header['pbc'] = parts[3:6]
            
            i += 1
            for _ in range(3):
                if i < len(lines):
                    bound_line = lines[i].strip()
                    bound_line = bound_line.replace('0.00.0', '0.0 0.0')
                    parts = bound_line.split()
                    if len(parts) >= 2:
                        lo, hi = float(parts[0]), float(parts[1])
                        header['box_bounds'].append([lo, hi])
                    i += 1
                    
        elif line.startswith("ITEM: ATOMS"):
            columns = line.split()[2:]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("ITEM:"):
                atom_line = lines[i].strip()
                parts = atom_line.split()
                if len(parts) >= len(columns):
                    atom_lines.append(atom_line + '\n')
                i += 1
        else:
            i += 1
    
    if not atom_lines:
        raise ValueError("No se encontraron datos de átomos en el archivo")
    
    if not header['box_bounds']:
        raise ValueError("No se encontraron límites de caja en el archivo")
    
    # Crear DataFrame
    data_io = StringIO("".join(atom_lines))
    df = pd.read_csv(data_io, sep=r'\s+', names=columns)
    
    return header, df


def write_lammps_dump(output_path: str, header: Dict[str, Any], df: pd.DataFrame):
    """Escribe archivo LAMMPS dump"""
    with open(output_path, 'w') as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{header['timestep']}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{len(df)}\n")
        f.write(f"ITEM: BOX BOUNDS {' '.join(header['pbc'])}\n")
        for bounds in header['box_bounds']:
            f.write(f"{bounds[0]:.6f} {bounds[1]:.6f}\n")
        f.write(f"ITEM: ATOMS {' '.join(df.columns)}\n")
        
        for _, row in df.iterrows():
            values = []
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    values.append("0")
                elif isinstance(val, (int, np.integer)):
                    values.append(str(int(val)))
                elif isinstance(val, (float, np.floating)):
                    values.append(f"{val:.8f}")
                else:
                    values.append(str(val))
            f.write(" ".join(values) + "\n")

# ==========================================
# CLUSTERING LOGIC
# ==========================================

class ClusteringEngine:
    """Motor de clustering unificado para múltiples algoritmos"""
    
    def __init__(self, data_tuple=None):
        self.header = None
        self.data_df = None
        self.coords = None
        self.labels = None
        self.metrics = {}
        
        if data_tuple:
            self.header, self.data_df = data_tuple

    def leer_dump(self):
        if self.data_df is None:
            raise ValueError("No hay datos cargados")
        self.coords = self.data_df[['x', 'y', 'z']].values
        return len(self.coords)
    
    def aplicar_kmeans(self, n_clusters=5):
        """KMeans clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = kmeans.fit_predict(self.coords)
        n_clusters_found = len(np.unique(self.labels))
        
        self.data_df['Cluster'] = self.labels
        self.data_df['Cluster_Level'] = 0
        
        # Calcular métricas
        self.metrics = {
            'silhouette': silhouette_score(self.coords, self.labels) if n_clusters_found > 1 else 0,
            'davies_bouldin': davies_bouldin_score(self.coords, self.labels) if n_clusters_found > 1 else 0,
            'calinski_harabasz': calinski_harabasz_score(self.coords, self.labels) if n_clusters_found > 1 else 0
        }
        
        return n_clusters_found
    
    def aplicar_meanshift(self, quantile=0.2):
        """MeanShift clustering"""
        bandwidth = estimate_bandwidth(
            self.coords, 
            quantile=quantile, 
            n_samples=min(500, len(self.coords))
        )
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        self.labels = ms.fit_predict(self.coords)
        n_clusters_found = len(np.unique(self.labels))
        
        self.data_df['Cluster'] = self.labels
        self.data_df['Cluster_Level'] = 0
        
        # Calcular métricas
        self.metrics = {
            'silhouette': silhouette_score(self.coords, self.labels) if n_clusters_found > 1 else 0,
            'davies_bouldin': davies_bouldin_score(self.coords, self.labels) if n_clusters_found > 1 else 0,
            'calinski_harabasz': calinski_harabasz_score(self.coords, self.labels) if n_clusters_found > 1 else 0
        }
        
        return n_clusters_found
    
    def aplicar_aglomerativo(self, n_clusters=5, linkage_method='ward'):
        """Clustering aglomerativo jerárquico"""
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        self.labels = agg.fit_predict(self.coords)
        n_clusters_found = len(np.unique(self.labels))
        
        self.data_df['Cluster'] = self.labels
        self.data_df['Cluster_Level'] = 0
        
        # Calcular métricas
        self.metrics = {
            'silhouette': silhouette_score(self.coords, self.labels) if n_clusters_found > 1 else 0,
            'davies_bouldin': davies_bouldin_score(self.coords, self.labels) if n_clusters_found > 1 else 0,
            'calinski_harabasz': calinski_harabasz_score(self.coords, self.labels) if n_clusters_found > 1 else 0
        }
        
        return n_clusters_found
    
    def aplicar_hdbscan(self, min_cluster_size=10, min_samples=None):
        """HDBSCAN clustering"""
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN no está instalado. Usa: pip install hdbscan")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        self.labels = clusterer.fit_predict(self.coords)
        n_clusters_found = len(np.unique(self.labels))
        
        self.data_df['Cluster'] = self.labels
        self.data_df['Cluster_Level'] = 0
        
        # Calcular métricas (excluyendo ruido -1)
        mask = self.labels != -1
        if mask.sum() > 0:
            self.metrics = {
                'silhouette': silhouette_score(self.coords[mask], self.labels[mask]) if n_clusters_found > 1 else 0,
                'davies_bouldin': davies_bouldin_score(self.coords[mask], self.labels[mask]) if n_clusters_found > 1 else 0,
                'calinski_harabasz': calinski_harabasz_score(self.coords[mask], self.labels[mask]) if n_clusters_found > 1 else 0,
                'noise_points': (self.labels == -1).sum()
            }
        
        return n_clusters_found

# ==========================================
# FUNCIONES DE VISUALIZACIÓN
# ==========================================

def generate_distinct_colors(n):
    """Genera 'n' colores distintos en formato hexadecimal"""
    if n <= 12:
        base_colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
            "#F7DC6F", "#BB8FCE", "#85C1E2", "#F8B88B", "#A8D5BA",
            "#E8B4B8", "#FFB6B9"
        ]
        return base_colors[:n]
    else:
        hues = np.linspace(0, 1, n, endpoint=False)
        colors = [colorsys.hsv_to_rgb(h, 0.8, 0.9) for h in hues]
        return ["#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]


def create_3d_clustering_viz(df, show_clusters=True, marker_size=5, show_noise=True):
    """Crea visualización 3D interactiva con Plotly para clusters"""
    
    fig = go.Figure()
    
    if show_clusters and 'Cluster' in df.columns:
        clusters = sorted([c for c in df['Cluster'].unique() if c != -1])
        
        # Mostrar clusters normales
        colors = generate_distinct_colors(len(clusters))
        
        for i, cluster_id in enumerate(clusters):
            cluster_data = df[df['Cluster'] == cluster_id]
            fig.add_trace(go.Scatter3d(
                x=cluster_data['x'],
                y=cluster_data['y'],
                z=cluster_data['z'],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(
                    size=marker_size,
                    color=colors[i],
                    opacity=0.8,
                    line=dict(width=0.5, color='white')
                ),
                text=[f"Cluster: {cluster_id}" for _ in range(len(cluster_data))],
                hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ))
        
        # Mostrar puntos de ruido (si existen y si está habilitado)
        if show_noise and (-1 in df['Cluster'].values):
            noise_data = df[df['Cluster'] == -1]
            fig.add_trace(go.Scatter3d(
                x=noise_data['x'],
                y=noise_data['y'],
                z=noise_data['z'],
                mode='markers',
                name='Ruido (sin cluster)',
                marker=dict(
                    size=marker_size,
                    color='gray',
                    opacity=0.4,
                    symbol='x',
                    line=dict(width=1, color='darkgray')
                ),
                text=["Ruido" for _ in range(len(noise_data))],
                hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ))
    
    fig.update_layout(
        title="Visualización 3D de Clusters",
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=1000,
        height=700,
        hovermode='closest',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig


def create_cluster_distribution_plot(df):
    """Crea gráfico de distribución de clusters"""
    if 'Cluster' not in df.columns:
        return None
    
    cluster_counts = df[df['Cluster'] != -1]['Cluster'].value_counts().sort_index()
    colors = generate_distinct_colors(len(cluster_counts))
    
    fig = go.Figure(data=[
        go.Bar(
            x=[f'Cluster {i}' for i in cluster_counts.index],
            y=cluster_counts.values,
            marker=dict(color=colors),
            text=cluster_counts.values,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Átomos: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Distribución de Átomos por Cluster",
        xaxis_title="Cluster",
        yaxis_title="Número de Átomos",
        height=400,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

# ==========================================
# INTERFAZ PRINCIPAL DE STREAMLIT
# ==========================================

# SIDEBAR - CARGA DE ARCHIVO
with st.sidebar:
    st.header("📤 Paso 1: Cargar Archivo")
    uploaded_file = st.file_uploader(
        "Selecciona archivo LAMMPS dump",
        type=['dump', 'txt'],
        help="Archivo .dump o .txt en formato LAMMPS"
    )

if uploaded_file is None:
    # Pantalla inicial
    st.info("👆 **Carga un archivo LAMMPS dump para comenzar el análisis**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Métodos de Clustering:
        - ✅ **KMeans** - Control manual del número de clusters
        - ✅ **MeanShift** - Estimación automática
        - ✅ **Aglomerativo** - Clustering jerárquico
        - ✅ **HDBSCAN** - Detección de densidad + ruido (requiere instalación)
        """)
    
    with col2:
        st.markdown("""
        ### 📁 Exportación Avanzada:
        - ✅ Archivo consolidado (.dump)
        - ✅ **Directorio con clusters individuales**
        - ✅ Cada cluster en archivo .dump separado
        - ✅ Ruido en archivo aparte (HDBSCAN)
        - ✅ Todo empaquetado en ZIP
        
        ### 📄 Formatos soportados:
        - **LAMMPS dump** (.dump, .txt)
        - **Columnas requeridas:** x, y, z
        """)
    
    # Mostrar advertencia si HDBSCAN no está disponible
    if not HDBSCAN_AVAILABLE:
        st.warning("""
        ⚠️ **HDBSCAN no está instalado**
        
        Para usar HDBSCAN, instala con:
        ```bash
        pip install hdbscan
        ```
        """)

else:
    # PROCESAR ARCHIVO
    try:
        file_content = uploaded_file.read()
        header, df = parse_lammps_dump(file_content)
        
        st.success(f"✅ Archivo cargado: **{uploaded_file.name}**")
        st.markdown("---")
        
        # SIDEBAR - PARÁMETROS
        with st.sidebar:
            st.header("⚙️ Paso 2: Configuración")
            
            clustering_method = st.radio(
                "Método de clustering:",
                ["KMeans", "MeanShift", "Aglomerativo", "HDBSCAN"] if HDBSCAN_AVAILABLE else ["KMeans", "MeanShift", "Aglomerativo"],
                help="Elige el algoritmo de clustering"
            )
            
            st.markdown("---")
            st.subheader("⚙️ Parámetros del Método")
            
            # Parámetros específicos por método
            if clustering_method == "KMeans":
                n_clusters = st.slider(
                    "Número de clusters",
                    min_value=2,
                    max_value=min(50, len(df) // 10),
                    value=5,
                    step=1
                )
                method_params = {'n_clusters': n_clusters}
            
            elif clustering_method == "MeanShift":
                quantile = st.slider(
                    "Quantile para estimación de ancho de banda",
                    min_value=0.05,
                    max_value=0.95,
                    value=0.2,
                    step=0.05
                )
                method_params = {'quantile': quantile}
            
            elif clustering_method == "Aglomerativo":
                n_clusters = st.slider(
                    "Número de clusters",
                    min_value=2,
                    max_value=min(50, len(df) // 10),
                    value=5,
                    step=1
                )
                linkage_method = st.selectbox(
                    "Método de linkage:",
                    ["ward", "complete", "average", "single"],
                    help="ward: minimiza varianza, complete: máx distancia, average: promedio"
                )
                method_params = {'n_clusters': n_clusters, 'linkage_method': linkage_method}
            
            elif clustering_method == "HDBSCAN":
                min_cluster_size = st.slider(
                    "Tamaño mínimo de cluster",
                    min_value=5,
                    max_value=100,
                    value=10,
                    step=5,
                    help="Número mínimo de puntos para formar un cluster válido"
                )
                min_samples = st.slider(
                    "Muestras mínimas",
                    min_value=1,
                    max_value=50,
                    value=min_cluster_size,
                    step=1,
                    help="Controla la separación de clusters (influencia local de densidad)"
                )
                method_params = {'min_cluster_size': min_cluster_size, 'min_samples': min_samples}
            
            # Parámetros avanzados compartidos
            st.markdown("---")
            st.subheader("🔧 Configuraciones Avanzadas")
            
            with st.expander("Mostrar/Ocultar opciones avanzadas", expanded=False):
                show_noise = st.checkbox("Mostrar puntos de ruido (HDBSCAN)", value=True)
                marker_size = st.slider("Tamaño de marcadores", 1, 15, 5, key="marker_size_advanced")
                enable_metrics = st.checkbox("Calcular métricas de calidad", value=True)
            
            st.header("🚀 Paso 3: Ejecutar")
            run_button = st.button("Iniciar Clustering", type="primary", use_container_width=True)
        
        # PASO 1: INFORMACIÓN DEL ARCHIVO
        st.header("📊 Paso 1: Información del Archivo")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de Átomos", f"{len(df):,}")
        col2.metric("Timestep", f"{header['timestep']}")
        col3.metric("Dimensión X", f"{header['box_bounds'][0][0]:.1f} - {header['box_bounds'][0][1]:.1f} Å")
        col4.metric("Dimensión Y", f"{header['box_bounds'][1][0]:.1f} - {header['box_bounds'][1][1]:.1f} Å")
        
        st.markdown("---")
        
        # PASO 2: PREVISUALIZACIÓN
        st.header("👀 Paso 2: Previsualización de Datos")
        
        col_prev1, col_prev2 = st.columns(2)
        
        with col_prev1:
            st.subheader("Primeras filas del archivo")
            st.dataframe(df.head(10), use_container_width=True, height=300)
        
        with col_prev2:
            st.subheader("Estadísticas de coordenadas")
            stats_df = df[['x', 'y', 'z']].describe()
            st.dataframe(stats_df, use_container_width=True)
        
        st.markdown("---")
        
        # PASO 3: EJECUTAR CLUSTERING
        if run_button:
            st.header("🔄 Paso 3: Procesando Clustering")
            
            with st.spinner("Realizando análisis de clustering..."):
                try:
                    # Inicializar engine
                    engine = ClusteringEngine(data_tuple=(header, df.copy()))
                    engine.leer_dump()
                    
                    # Aplicar clustering según método
                    if clustering_method == "KMeans":
                        n_clusters_found = engine.aplicar_kmeans(**method_params)
                    elif clustering_method == "MeanShift":
                        n_clusters_found = engine.aplicar_meanshift(**method_params)
                    elif clustering_method == "Aglomerativo":
                        n_clusters_found = engine.aplicar_aglomerativo(**method_params)
                    elif clustering_method == "HDBSCAN":
                        n_clusters_found = engine.aplicar_hdbscan(**method_params)
                    
                    # Guardar resultado en session_state
                    st.session_state['clustering_result'] = {
                        'df': engine.data_df,
                        'header': header,
                        'n_clusters': n_clusters_found,
                        'method': clustering_method,
                        'metrics': engine.metrics,
                        'params': method_params
                    }
                    
                    st.success(f"✅ Clustering completado: **{n_clusters_found} clusters** detectados")
                    
                except Exception as e:
                    st.error(f"❌ Error durante clustering: {str(e)}")
                    with st.expander("Ver detalles del error"):
                        st.code(traceback.format_exc())
        
        st.markdown("---")
        
        # PASO 4: RESULTADOS
        if 'clustering_result' in st.session_state:
            st.header("📈 Paso 4: Resultados del Análisis")
            
            result = st.session_state['clustering_result']
            clustered_df = result['df']
            
            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(
                "Clusters Encontrados",
                f"{result['n_clusters']}",
                delta=None
            )
            
            col2.metric(
                "Método",
                result['method'],
                delta=None
            )
            
            cluster_sizes = clustered_df[clustered_df['Cluster'] != -1]['Cluster'].value_counts()
            if len(cluster_sizes) > 0:
                col3.metric(
                    "Cluster más grande",
                    f"{cluster_sizes.max():,} átomos",
                    delta=None
                )
                
                col4.metric(
                    "Cluster más pequeño",
                    f"{cluster_sizes.min():,} átomos",
                    delta=None
                )
            
            # Mostrar puntos de ruido si es HDBSCAN
            if result['method'] == 'HDBSCAN':
                noise_count = (clustered_df['Cluster'] == -1).sum()
                if noise_count > 0:
                    st.info(f"👻 Puntos de ruido (sin asignar a cluster): **{noise_count:,}** ({100*noise_count/len(clustered_df):.1f}%)")
            
            st.markdown("---")
            
            # Mostrar métricas de calidad si está habilitado
            if enable_metrics and result['metrics']:
                st.subheader("📊 Métricas de Calidad de Clustering")
                
                metrics_cols = st.columns(3)
                
                if 'silhouette' in result['metrics']:
                    metrics_cols[0].metric(
                        "Silhouette Score",
                        f"{result['metrics']['silhouette']:.3f}",
                        help="Rango [-1, 1]. Valores altos indican clusters bien definidos"
                    )
                
                if 'davies_bouldin' in result['metrics']:
                    metrics_cols[1].metric(
                        "Davies-Bouldin Index",
                        f"{result['metrics']['davies_bouldin']:.3f}",
                        help="Valores bajos indican mejor separación de clusters"
                    )
                
                if 'calinski_harabasz' in result['metrics']:
                    metrics_cols[2].metric(
                        "Calinski-Harabasz Index",
                        f"{result['metrics']['calinski_harabasz']:.3f}",
                        help="Valores altos indican clusters densos y bien separados"
                    )
                
                if 'noise_points' in result['metrics']:
                    st.metric(
                        "Puntos de ruido detectados",
                        result['metrics']['noise_points']
                    )
                
                st.markdown("---")
            
            # VISUALIZACIÓN 3D
            st.subheader("🎨 Visualización 3D Interactiva")
            
            col_viz1, col_viz2, col_viz3 = st.columns(3)
            
            with col_viz1:
                show_clusters = st.checkbox("Mostrar clusters", value=True)
            
            with col_viz2:
                show_noise_viz = st.checkbox("Mostrar ruido", value=True)
            
            with col_viz3:
                marker_size_final = st.slider("Tamaño de marcadores", 1, 15, 5, key="marker_size_final")
            
            if show_clusters:
                fig = create_3d_clustering_viz(clustered_df, marker_size=marker_size_final, show_noise=show_noise_viz)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # DISTRIBUCIÓN DE CLUSTERS
            st.subheader("📊 Distribución de Clusters")
            
            dist_fig = create_cluster_distribution_plot(clustered_df)
            if dist_fig:
                st.plotly_chart(dist_fig, use_container_width=True)
            
            # Tabla de estadísticas
            with st.expander("📋 Estadísticas detalladas por cluster", expanded=True):
                cluster_stats = clustered_df[clustered_df['Cluster'] != -1].groupby('Cluster').agg({
                    'x': ['min', 'max', 'mean'],
                    'y': ['min', 'max', 'mean'],
                    'z': ['min', 'max', 'mean'],
                    'id': 'count'
                }).round(2)
                cluster_stats.columns = ['X_min', 'X_max', 'X_mean', 'Y_min', 'Y_max', 'Y_mean', 'Z_min', 'Z_max', 'Z_mean', 'N_Átomos']
                st.dataframe(cluster_stats, use_container_width=True)
            
            # Mostrar parámetros utilizados
            with st.expander("🔧 Parámetros utilizados", expanded=False):
                st.json(result['params'])
            
            st.markdown("---")
            
            # ==========================================
            # EXPORTACIÓN AVANZADA CON DIRECTORIOS
            # ==========================================
            st.header("💾 Paso 5: Exportación de Resultados")
            
            # Opciones de exportación
            export_type = st.radio(
                "Tipo de exportación:",
                ["📁 Directorio con clusters + Archivo consolidado", "📁 Solo directorio con clusters", "📄 Solo archivo consolidado"],
                help="Elige qué deseas exportar"
            )
            
            st.markdown("---")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                output_dirname = st.text_input(
                    "Nombre del directorio",
                    value=f"clusters_{Path(uploaded_file.name).stem}",
                    help="Nombre para el directorio de exportación"
                )
            
            with col2:
                st.write("")
                st.write("")
                export_button = st.button("📥 Exportar", type="primary", use_container_width=True)
            
            if export_button:
                try:
                    # Crear directorio temporal
                    temp_export_dir = f"/tmp/{output_dirname}"
                    os.makedirs(temp_export_dir, exist_ok=True)
                    
                    with st.spinner("Preparando archivos para exportación..."):
                        
                        # 1. EXPORTAR ARCHIVO CONSOLIDADO (opcional)
                        if export_type in ["📁 Directorio con clusters + Archivo consolidado", "📄 Solo archivo consolidado"]:
                            consolidated_filename = f"{output_dirname}_consolidated.dump"
                            consolidated_path = os.path.join(temp_export_dir, consolidated_filename)
                            write_lammps_dump(consolidated_path, header, clustered_df)
                            st.info(f"✅ Archivo consolidado creado: **{consolidated_filename}**")
                        
                        # 2. EXPORTAR CLUSTERS INDIVIDUALES
                        if export_type in ["📁 Directorio con clusters + Archivo consolidado", "📁 Solo directorio con clusters"]:
                            st.write("")
                            st.write("📁 **Exportando clusters individuales:**")
                            
                            cluster_files = []
                            clusters_to_export = sorted([c for c in clustered_df['Cluster'].unique() if c != -1])
                            
                            progress_bar = st.progress(0)
                            
                            for idx, cluster_id in enumerate(clusters_to_export):
                                cluster_data = clustered_df[clustered_df['Cluster'] == cluster_id].copy()
                                cluster_filename = f"cluster_{cluster_id:03d}_{len(cluster_data)}_atoms.dump"
                                cluster_path = os.path.join(temp_export_dir, cluster_filename)
                                
                                # Escribir archivo del cluster
                                write_lammps_dump(cluster_path, header, cluster_data)
                                cluster_files.append((cluster_id, cluster_filename, len(cluster_data)))
                                
                                st.write(f"  • Cluster {cluster_id}: **{len(cluster_data):,}** átomos")
                                progress_bar.progress((idx + 1) / len(clusters_to_export))
                            
                            st.success(f"✅ **{len(clusters_to_export)}** archivos de clusters creados")
                            
                            # Exportar ruido si existe y es HDBSCAN
                            if result['method'] == 'HDBSCAN' and (-1 in clustered_df['Cluster'].values):
                                noise_data = clustered_df[clustered_df['Cluster'] == -1].copy()
                                noise_filename = f"noise_{len(noise_data)}_atoms.dump"
                                noise_path = os.path.join(temp_export_dir, noise_filename)
                                write_lammps_dump(noise_path, header, noise_data)
                                st.write(f"  • 👻 Ruido: **{len(noise_data):,}** átomos")
                        
                        # 3. CREAR ZIP
                        st.write("")
                        st.write("📦 **Comprimiendo archivos...**")
                        
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for root, dirs, files in os.walk(temp_export_dir):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arcname = os.path.relpath(file_path, temp_export_dir)
                                    zip_file.write(file_path, arcname)
                        
                        zip_buffer.seek(0)
                        
                        st.markdown("---")
                        st.download_button(
                            label="⬇️ Descargar ZIP",
                            data=zip_buffer.getvalue(),
                            file_name=f"{output_dirname}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                        
                        st.success(f"✅ **{output_dirname}.zip** listo para descargar")
                        
                        with st.expander("📋 Resumen de Exportación", expanded=True):
                            st.markdown(f"""
                            ### Estadísticas Finales:
                            - **Átomos totales:** {len(clustered_df):,}
                            - **Clusters generados:** {result['n_clusters']}
                            - **Método de clustering:** {result['method']}
                            - **Átomos por cluster (promedio):** {len([c for c in clustered_df['Cluster'].unique() if c != -1]) and len(clustered_df[clustered_df['Cluster'] != -1]) / len([c for c in clustered_df['Cluster'].unique() if c != -1]):.0f}
                            - **Tamaño de cluster (máx):** {cluster_sizes.max():,} átomos
                            - **Tamaño de cluster (mín):** {cluster_sizes.min():,} átomos
                            
                            ### Contenido del ZIP:
                            """)
                            
                            if export_type in ["📁 Directorio con clusters + Archivo consolidado", "📄 Solo archivo consolidado"]:
                                st.markdown(f"- **{output_dirname}_consolidated.dump** (archivo con todos los clusters)")
                            
                            if export_type in ["📁 Directorio con clusters + Archivo consolidado", "📁 Solo directorio con clusters"]:
                                st.markdown(f"- **{len(clusters_to_export)} archivos** de clusters individuales")
                                if result['method'] == 'HDBSCAN' and (-1 in clustered_df['Cluster'].values):
                                    st.markdown("- **noise_XXXX_atoms.dump** (archivo con puntos de ruido)")
                        
                        # Limpiar directorio temporal
                        shutil.rmtree(temp_export_dir, ignore_errors=True)
                    
                except Exception as e:
                    st.error(f"❌ Error al exportar: {str(e)}")
                    with st.expander("Ver detalles del error"):
                        st.code(traceback.format_exc())
    
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {str(e)}")
        with st.expander("Ver detalles del error"):
            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🧩 Clustering Avanzado | Advanced Analysis Tool | "
    "Powered by Python + Streamlit + Scikit-learn + HDBSCAN | "
    "📁 Exportación de Clusters Individuales"
    "</div>",
    unsafe_allow_html=True
)