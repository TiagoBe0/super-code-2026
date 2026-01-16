"""
🧩 CLUSTERING HDBSCAN AVANZADO
Análisis con control de tamaño mínimo y máximo de clusters
Exportación individual de clusters en archivos .dump separados
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import shutil
import traceback
from pathlib import Path
from io import StringIO, BytesIO
from typing import Dict, List, Tuple, Any
import zipfile
import colorsys
import plotly.graph_objects as go

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    st.error("❌ HDBSCAN no está instalado. Ejecuta: pip install hdbscan")
    st.stop()

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="HDBSCAN Clustering",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧩 Clustering HDBSCAN con Control de Tamaño</div>', unsafe_allow_html=True)
st.markdown("**Análisis automático con restricciones de tamaño mínimo y máximo por cluster**")
st.markdown("---")

# ==========================================
# FUNCIONES DE PARSEO LAMMPS
# ==========================================

@st.cache_data
def parse_lammps_dump(file_content: bytes) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Parser robusto para archivos LAMMPS dump"""
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
# CLUSTERING CON CONTROL DE TAMAÑO
# ==========================================

class HDBSCANClusterer:
    """Motor de clustering HDBSCAN con control de tamaño de clusters"""
    
    def __init__(self, data_tuple):
        self.header, self.data_df = data_tuple
        self.coords = None
        self.labels = None
        self.probabilities = None
        self.outlier_scores = None
        
    def aplicar_clustering(self, min_cluster_size=10, min_samples=None, 
                          min_atoms_per_cluster=None, max_atoms_per_cluster=None):
        """
        Aplica HDBSCAN con control de tamaño de clusters
        
        Args:
            min_cluster_size: Tamaño mínimo para que se forme un cluster
            min_samples: Controla densidad local (None = usa min_cluster_size)
            min_atoms_per_cluster: Filtrar clusters con menos átomos que este valor
            max_atoms_per_cluster: Dividir clusters con más átomos que este valor
        """
        self.coords = self.data_df[['x', 'y', 'z']].values
        
        # Aplicar HDBSCAN inicial
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method='eom',  # Excess of Mass
            prediction_data=True
        )
        
        self.labels = clusterer.fit_predict(self.coords)
        self.probabilities = clusterer.probabilities_
        self.outlier_scores = clusterer.outlier_scores_
        
        # Aplicar restricciones de tamaño
        if min_atoms_per_cluster or max_atoms_per_cluster:
            self.labels = self._apply_size_constraints(
                self.labels, 
                min_atoms_per_cluster, 
                max_atoms_per_cluster
            )
        
        n_clusters = len([l for l in np.unique(self.labels) if l != -1])
        
        self.data_df['Cluster'] = self.labels
        self.data_df['Probability'] = self.probabilities
        self.data_df['Outlier_Score'] = self.outlier_scores
        
        return n_clusters
    
    def _apply_size_constraints(self, labels, min_atoms, max_atoms):
        """Aplica restricciones de tamaño mínimo y máximo"""
        new_labels = labels.copy()
        
        unique_labels = [l for l in np.unique(labels) if l != -1]
        
        # Filtrar clusters pequeños
        if min_atoms:
            for label in unique_labels:
                cluster_size = (labels == label).sum()
                if cluster_size < min_atoms:
                    # Marcar como ruido
                    new_labels[labels == label] = -1
        
        # Subdividir clusters grandes
        if max_atoms:
            next_label = max(unique_labels) + 1 if unique_labels else 0
            
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_size = cluster_mask.sum()
                
                if cluster_size > max_atoms:
                    # Subdividir usando K-means
                    cluster_coords = self.coords[cluster_mask]
                    n_splits = int(np.ceil(cluster_size / max_atoms))
                    
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
                    sub_labels = kmeans.fit_predict(cluster_coords)
                    
                    # Asignar nuevas etiquetas
                    indices = np.where(cluster_mask)[0]
                    for i, sub_label in enumerate(sub_labels):
                        new_labels[indices[i]] = next_label + sub_label
                    
                    next_label += n_splits
        
        return new_labels
    
    def get_cluster_stats(self):
        """Obtiene estadísticas de los clusters"""
        if self.labels is None:
            return None
        
        stats = []
        unique_labels = sorted([l for l in np.unique(self.labels) if l != -1])
        
        for label in unique_labels:
            cluster_mask = self.labels == label
            cluster_data = self.data_df[cluster_mask]
            
            stats.append({
                'Cluster': label,
                'N_Átomos': cluster_mask.sum(),
                'X_min': cluster_data['x'].min(),
                'X_max': cluster_data['x'].max(),
                'Y_min': cluster_data['y'].min(),
                'Y_max': cluster_data['y'].max(),
                'Z_min': cluster_data['z'].min(),
                'Z_max': cluster_data['z'].max(),
                'Prob_Media': cluster_data['Probability'].mean(),
                'Outlier_Score_Media': cluster_data['Outlier_Score'].mean()
            })
        
        return pd.DataFrame(stats)

# ==========================================
# VISUALIZACIÓN
# ==========================================

def generate_distinct_colors(n):
    """Genera colores distintos"""
    if n <= 12:
        return ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
                "#F7DC6F", "#BB8FCE", "#85C1E2", "#F8B88B", "#A8D5BA",
                "#E8B4B8", "#FFB6B9"][:n]
    else:
        hues = np.linspace(0, 1, n, endpoint=False)
        colors = [colorsys.hsv_to_rgb(h, 0.8, 0.9) for h in hues]
        return ["#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255)) for r, g, b in colors]


def create_3d_viz(df, marker_size=5, show_noise=True, color_by='cluster'):
    """Crea visualización 3D interactiva"""
    fig = go.Figure()
    
    if 'Cluster' not in df.columns:
        return fig
    
    clusters = sorted([c for c in df['Cluster'].unique() if c != -1])
    colors = generate_distinct_colors(len(clusters))
    
    # Clusters normales
    for i, cluster_id in enumerate(clusters):
        cluster_data = df[df['Cluster'] == cluster_id]
        
        if color_by == 'probability':
            color_scale = cluster_data['Probability']
            colorscale = 'Viridis'
        else:
            color_scale = colors[i]
            colorscale = None
        
        fig.add_trace(go.Scatter3d(
            x=cluster_data['x'],
            y=cluster_data['y'],
            z=cluster_data['z'],
            mode='markers',
            name=f'Cluster {cluster_id} ({len(cluster_data)} átomos)',
            marker=dict(
                size=marker_size,
                color=color_scale,
                colorscale=colorscale,
                opacity=0.8,
                line=dict(width=0.5, color='white')
            ),
            text=[f"Cluster: {cluster_id}<br>Prob: {p:.3f}" 
                  for p in cluster_data['Probability']],
            hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
        ))
    
    # Ruido
    if show_noise and (-1 in df['Cluster'].values):
        noise_data = df[df['Cluster'] == -1]
        fig.add_trace(go.Scatter3d(
            x=noise_data['x'],
            y=noise_data['y'],
            z=noise_data['z'],
            mode='markers',
            name=f'Ruido ({len(noise_data)} átomos)',
            marker=dict(
                size=marker_size,
                color='gray',
                opacity=0.3,
                symbol='x'
            ),
            hovertemplate='<b>Ruido</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Visualización 3D de Clusters HDBSCAN",
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        width=1000,
        height=700,
        hovermode='closest'
    )
    
    return fig


def create_distribution_plot(df):
    """Gráfico de distribución de clusters"""
    if 'Cluster' not in df.columns:
        return None
    
    cluster_counts = df[df['Cluster'] != -1]['Cluster'].value_counts().sort_index()
    colors = generate_distinct_colors(len(cluster_counts))
    
    fig = go.Figure(data=[
        go.Bar(
            x=[f'C{i}' for i in cluster_counts.index],
            y=cluster_counts.values,
            marker=dict(color=colors),
            text=cluster_counts.values,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Distribución de Átomos por Cluster",
        xaxis_title="Cluster",
        yaxis_title="Número de Átomos",
        height=400,
        showlegend=False
    )
    
    return fig

# ==========================================
# INTERFAZ STREAMLIT
# ==========================================

# SIDEBAR
with st.sidebar:
    st.header("📤 Paso 1: Cargar Archivo")
    uploaded_file = st.file_uploader(
        "Archivo LAMMPS dump",
        type=['dump', 'txt'],
        help="Archivo .dump en formato LAMMPS"
    )
    
    if uploaded_file:
        st.success("✅ Archivo cargado")

# Pantalla inicial
if uploaded_file is None:
    st.info("👆 **Carga un archivo LAMMPS dump para comenzar**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Características:
        - ✅ Clustering HDBSCAN optimizado
        - ✅ Control de tamaño mínimo de clusters
        - ✅ Control de tamaño máximo de clusters
        - ✅ Detección automática de ruido
        - ✅ Métricas de probabilidad y outliers
        """)
    
    with col2:
        st.markdown("""
        ### 📦 Exportación:
        - ✅ Archivo consolidado con todos los clusters
        - ✅ Clusters individuales en archivos .dump separados
        - ✅ Archivo separado para puntos de ruido
        - ✅ Estadísticas detalladas en CSV
        - ✅ Todo empaquetado en ZIP
        """)
    
    st.stop()

# PROCESAR ARCHIVO
try:
    file_content = uploaded_file.read()
    header, df = parse_lammps_dump(file_content)
    
    st.success(f"✅ Archivo cargado: **{uploaded_file.name}**")
    st.markdown("---")
    
    # SIDEBAR - PARÁMETROS
    with st.sidebar:
        st.header("⚙️ Paso 2: Parámetros HDBSCAN")
        
        st.subheader("🔧 Parámetros Básicos")
        min_cluster_size = st.slider(
            "Tamaño mínimo para formar cluster",
            min_value=5,
            max_value=200,
            value=15,
            step=5,
            help="Número mínimo de puntos para considerar un cluster válido"
        )
        
        min_samples = st.slider(
            "Muestras mínimas (densidad local)",
            min_value=1,
            max_value=50,
            value=min_cluster_size,
            step=1,
            help="Controla qué tan densa debe ser una región. Usa el mismo valor que min_cluster_size para mejor balance"
        )
        
        st.markdown("---")
        st.subheader("📏 Control de Tamaño de Clusters")
        
        enable_min_filter = st.checkbox(
            "Activar filtro de tamaño mínimo",
            value=False,
            help="Eliminar clusters con menos átomos que el mínimo especificado"
        )
        
        min_atoms_per_cluster = None
        if enable_min_filter:
            min_atoms_per_cluster = st.number_input(
                "Mínimo de átomos por cluster",
                min_value=1,
                max_value=1000,
                value=50,
                step=10,
                help="Clusters con menos átomos serán marcados como ruido"
            )
        
        enable_max_filter = st.checkbox(
            "Activar límite de tamaño máximo",
            value=False,
            help="Subdividir clusters que excedan el máximo especificado"
        )
        
        max_atoms_per_cluster = None
        if enable_max_filter:
            max_atoms_per_cluster = st.number_input(
                "Máximo de átomos por cluster",
                min_value=10,
                max_value=10000,
                value=500,
                step=50,
                help="Clusters más grandes se subdividirán automáticamente"
            )
        
        st.markdown("---")
        st.header("🚀 Paso 3: Ejecutar")
        run_button = st.button("Iniciar Clustering", type="primary", use_container_width=True)
    
    # INFORMACIÓN DEL ARCHIVO
    st.header("📊 Información del Archivo")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Átomos", f"{len(df):,}")
    col2.metric("Timestep", f"{header['timestep']}")
    col3.metric("Box X", f"{header['box_bounds'][0][1] - header['box_bounds'][0][0]:.1f} Å")
    col4.metric("Box Y", f"{header['box_bounds'][1][1] - header['box_bounds'][1][0]:.1f} Å")
    
    st.markdown("---")
    
    # PREVISUALIZACIÓN
    with st.expander("👀 Previsualización de datos", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            st.dataframe(df[['x', 'y', 'z']].describe(), use_container_width=True)
    
    # EJECUTAR CLUSTERING
    if run_button:
        st.header("🔄 Procesando Clustering")
        
        with st.spinner("Aplicando HDBSCAN..."):
            try:
                clusterer = HDBSCANClusterer(data_tuple=(header, df.copy()))
                
                n_clusters = clusterer.aplicar_clustering(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    min_atoms_per_cluster=min_atoms_per_cluster,
                    max_atoms_per_cluster=max_atoms_per_cluster
                )
                
                st.session_state['clustering_result'] = {
                    'clusterer': clusterer,
                    'header': header,
                    'n_clusters': n_clusters,
                    'params': {
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'min_atoms_per_cluster': min_atoms_per_cluster,
                        'max_atoms_per_cluster': max_atoms_per_cluster
                    }
                }
                
                st.success(f"✅ Clustering completado: **{n_clusters} clusters** detectados")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("Detalles del error"):
                    st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # RESULTADOS
    if 'clustering_result' in st.session_state:
        st.header("📈 Resultados del Análisis")
        
        result = st.session_state['clustering_result']
        clusterer = result['clusterer']
        clustered_df = clusterer.data_df
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        n_noise = (clustered_df['Cluster'] == -1).sum()
        cluster_sizes = clustered_df[clustered_df['Cluster'] != -1]['Cluster'].value_counts()
        
        col1.metric("Clusters", f"{result['n_clusters']}")
        col2.metric("Átomos Clusterizados", f"{len(clustered_df) - n_noise:,}")
        col3.metric("Ruido", f"{n_noise:,} ({100*n_noise/len(clustered_df):.1f}%)")
        
        if len(cluster_sizes) > 0:
            col4.metric("Tamaño Promedio", f"{cluster_sizes.mean():.0f} átomos")
        
        st.markdown("---")
        
        # Estadísticas detalladas
        st.subheader("📊 Estadísticas por Cluster")
        stats_df = clusterer.get_cluster_stats()
        if stats_df is not None:
            st.dataframe(stats_df, use_container_width=True, height=300)
        
        st.markdown("---")
        
        # Visualización 3D
        st.subheader("🎨 Visualización 3D")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            show_noise = st.checkbox("Mostrar ruido", value=True)
        with col2:
            marker_size = st.slider("Tamaño marcadores", 1, 15, 5)
        with col3:
            color_by = st.selectbox("Colorear por:", ["cluster", "probability"])
        
        fig = create_3d_viz(clustered_df, marker_size, show_noise, color_by)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Distribución
        st.subheader("📊 Distribución de Tamaños")
        dist_fig = create_distribution_plot(clustered_df)
        if dist_fig:
            st.plotly_chart(dist_fig, use_container_width=True)
        
        st.markdown("---")
        
        # EXPORTACIÓN
        st.header("💾 Exportación de Resultados")
        
        output_dir = st.text_input(
            "Nombre del directorio",
            value=f"hdbscan_{Path(uploaded_file.name).stem}"
        )
        
        export_button = st.button("📥 Exportar Todo", type="primary", use_container_width=True)
        
        if export_button:
            try:
                temp_dir = f"/tmp/{output_dir}"
                os.makedirs(temp_dir, exist_ok=True)
                
                with st.spinner("Preparando exportación..."):
                    # 1. Archivo consolidado
                    consolidated_path = os.path.join(temp_dir, f"{output_dir}_consolidated.dump")
                    write_lammps_dump(consolidated_path, header, clustered_df)
                    
                    # 2. Clusters individuales
                    st.write("📁 Exportando clusters individuales...")
                    clusters_to_export = sorted([c for c in clustered_df['Cluster'].unique() if c != -1])
                    
                    progress_bar = st.progress(0)
                    for idx, cluster_id in enumerate(clusters_to_export):
                        cluster_data = clustered_df[clustered_df['Cluster'] == cluster_id]
                        filename = f"cluster_{cluster_id:03d}_{len(cluster_data)}_atoms.dump"
                        filepath = os.path.join(temp_dir, filename)
                        write_lammps_dump(filepath, header, cluster_data)
                        progress_bar.progress((idx + 1) / len(clusters_to_export))
                    
                    # 3. Ruido
                    if n_noise > 0:
                        noise_data = clustered_df[clustered_df['Cluster'] == -1]
                        noise_path = os.path.join(temp_dir, f"noise_{n_noise}_atoms.dump")
                        write_lammps_dump(noise_path, header, noise_data)
                    
                    # 4. Estadísticas
                    if stats_df is not None:
                        stats_path = os.path.join(temp_dir, "cluster_statistics.csv")
                        stats_df.to_csv(stats_path, index=False)
                    
                    # 5. Crear ZIP
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for root, dirs, files in os.walk(temp_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, temp_dir)
                                zf.write(file_path, arcname)
                    
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="⬇️ Descargar ZIP Completo",
                        data=zip_buffer.getvalue(),
                        file_name=f"{output_dir}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                    
                    st.success(f"✅ **{output_dir}.zip** listo para descargar")
                    
                    with st.expander("📋 Resumen de Exportación"):
                        st.markdown(f"""
                        ### Archivos generados:
                        - 📄 `{output_dir}_consolidated.dump` - Archivo consolidado
                        - 📁 {len(clusters_to_export)} archivos de clusters individuales
                        - 👻 `noise_{n_noise}_atoms.dump` - Puntos de ruido
                        - 📊 `cluster_statistics.csv` - Estadísticas detalladas
                        
                        ### Parámetros utilizados:
                        - Min cluster size: {result['params']['min_cluster_size']}
                        - Min samples: {result['params']['min_samples']}
                        - Min átomos/cluster: {result['params']['min_atoms_per_cluster'] or 'No aplicado'}
                        - Max átomos/cluster: {result['params']['max_atoms_per_cluster'] or 'No aplicado'}
                        """)
                    
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
            except Exception as e:
                st.error(f"❌ Error al exportar: {str(e)}")
                with st.expander("Detalles"):
                    st.code(traceback.format_exc())

except Exception as e:
    st.error(f"❌ Error al procesar archivo: {str(e)}")
    with st.expander("Detalles del error"):
        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🧩 HDBSCAN Clustering | Control Avanzado de Tamaño | "
    "Powered by Python + Streamlit + HDBSCAN"
    "</div>",
    unsafe_allow_html=True
)
