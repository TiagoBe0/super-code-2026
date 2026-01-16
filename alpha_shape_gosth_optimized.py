"""
🧬 NANOPORE DETECTOR - Alpha Shape Surface Constructor (Optimizado FINAL)
Interfaz visual para detección de átomos en superficies de nanoporos
usando Ghost Particles + Alpha Shape

VERSIÓN FINAL - Corregida con interfaz correcta
"""

import streamlit as st
import pandas as pd
import numpy as np
import traceback
from pathlib import Path
from io import StringIO
from typing import Dict, Tuple, Any
import plotly.graph_objects as go

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Nanopore Detector - Alpha Shape",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧬 Nanopore Detector - Ghost Particles Analysis</div>', unsafe_allow_html=True)
st.markdown("**Detección automática de átomos en superficies de nanoporos mediante Ghost Particles**")
st.markdown("---")

# ==========================================
# IMPORTAR MÓDULOS ALPHA SHAPE
# ==========================================

ALPHA_SHAPE_GHOST_AVAILABLE = False

try:
    import sys
    sys.path.insert(0, '/mnt/project')
    sys.path.insert(0, '/mnt/user-data/uploads')
    sys.path.insert(0, '.')
    
    from alpha_shape_ghost_particles import AlphaShapeWithGhosts
    ALPHA_SHAPE_GHOST_AVAILABLE = True
    
except ImportError as e:
    st.error(f"⚠️ No se pudo importar AlphaShapeWithGhosts. Error: {e}")
    st.info("Asegúrate que el archivo 'alpha_shape_ghost_particles.py' esté disponible")

# ==========================================
# FUNCIONES OPTIMIZADAS
# ==========================================

@st.cache_data
def parse_lammps_dump(file_content: bytes) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Parser LAMMPS optimizado"""
    try:
        text = file_content.decode('utf-8')
    except UnicodeDecodeError:
        text = file_content.decode('latin-1')
    
    lines = text.split('\n')
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
            bounds_data = []
            for _ in range(3):
                if i < len(lines):
                    bound_line = lines[i].strip().replace('0.00.0', '0.0 0.0')
                    parts = bound_line.split()
                    if len(parts) >= 2:
                        bounds_data.append([float(parts[0]), float(parts[1])])
                    i += 1
            header['box_bounds'] = bounds_data
                    
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
    df = pd.read_csv(data_io, sep=r'\s+', names=columns, engine='c')
    
    return header, df


def write_lammps_dump_optimized(output_path: str, header: Dict[str, Any], df: pd.DataFrame):
    """Escritura optimizada a LAMMPS dump"""
    
    lines = [
        "ITEM: TIMESTEP\n",
        f"{header['timestep']}\n",
        "ITEM: NUMBER OF ATOMS\n",
        f"{len(df)}\n",
        f"ITEM: BOX BOUNDS {' '.join(header['pbc'])}\n"
    ]
    
    for bounds in header['box_bounds']:
        lines.append(f"{bounds[0]:.6f} {bounds[1]:.6f}\n")
    
    lines.append(f"ITEM: ATOMS {' '.join(df.columns)}\n")
    
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
        lines.append(" ".join(values) + "\n")
    
    with open(output_path, 'w') as f:
        f.writelines(lines)


def create_3d_visualization(original_df, filtered_df, show_bulk, show_surface, marker_size):
    """Crea visualización 3D interactiva"""
    traces = []
    
    if show_bulk and len(original_df) > len(filtered_df):
        # Obtener índices de bulk (los que NO están en filtered_df)
        # Como filtered_df se reset, usamos posiciones
        surface_indices = set()
        for idx in filtered_df.index:
            # filtered_df tiene índices 0, 1, 2... después de reset_index
            # Necesitamos mapear de vuelta a original_df
            pass
        
        # Alternativa más simple: usar position based
        n_total = len(original_df)
        n_surface = len(filtered_df)
        
        if n_surface < n_total:
            # Crear máscara de bulk
            bulk_indices = [i for i in range(n_total) if i not in range(n_surface)]
            
            # Mejor: comparar con surface_atoms_indices original
            traces.append(
                go.Scatter3d(
                    x=original_df['x'].values[:n_total-n_surface] if n_total > n_surface else [],
                    y=original_df['y'].values[:n_total-n_surface] if n_total > n_surface else [],
                    z=original_df['z'].values[:n_total-n_surface] if n_total > n_surface else [],
                    mode='markers',
                    name='Bulk Atoms',
                    marker=dict(
                        size=marker_size,
                        color='lightblue',
                        opacity=0.6,
                        line=dict(width=0)
                    )
                )
            )
    
    if show_surface and len(filtered_df) > 0:
        traces.append(
            go.Scatter3d(
                x=filtered_df['x'].values,
                y=filtered_df['y'].values,
                z=filtered_df['z'].values,
                mode='markers',
                name='Nanopore Atoms',
                marker=dict(
                    size=marker_size + 1,
                    color='red',
                    opacity=0.9,
                    line=dict(width=0)
                )
            )
        )
    
    if not traces:
        return None
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        title="3D Atomic Structure",
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)",
            aspectmode='data'
        ),
        height=600,
        hovermode='closest',
        template='plotly_dark'
    )
    
    return fig


def create_distribution_plot(original_df, filtered_df):
    """Gráfico de distribución en eje Z"""
    fig = go.Figure()
    
    fig.add_trace(
        go.Histogram(
            x=original_df['z'],
            name='All Atoms',
            opacity=0.6,
            nbinsx=50
        )
    )
    
    if len(filtered_df) > 0:
        fig.add_trace(
            go.Histogram(
                x=filtered_df['z'],
                name='Nanopore Atoms',
                opacity=0.8,
                nbinsx=50
            )
        )
    
    fig.update_layout(
        title="Z-axis Distribution",
        xaxis_title="Z Position (Å)",
        yaxis_title="Count",
        barmode='overlay',
        height=400,
        template='plotly_dark'
    )
    
    return fig


# ==========================================
# INICIALIZACIÓN SESSION STATE
# ==========================================

if 'alpha_result' not in st.session_state:
    st.session_state['alpha_result'] = None

# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================

if not ALPHA_SHAPE_GHOST_AVAILABLE:
    st.error("❌ Módulo AlphaShapeWithGhosts no disponible. Por favor instálalo.")
    st.stop()

uploaded_file = st.file_uploader(
    "📁 Carga un archivo LAMMPS dump",
    type=['dump', 'txt'],
    help="Formato LAMMPS dump con columnas x, y, z"
)

if uploaded_file is not None:
    try:
        # Leer archivo
        file_content = uploaded_file.read()
        header, df = parse_lammps_dump(file_content)
        
        st.markdown("---")
        st.header("📝 Paso 1: Información del Archivo")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Atoms", f"{len(df):,}")
        col2.metric("Timestep", header['timestep'])
        col3.metric("Box Bounds", f"{len(header['box_bounds'])} dims")
        col4.metric("Columns", len(df.columns))
        
        st.markdown("---")
        st.header("⚙️ Paso 2: Parámetros Ghost Particles")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            probe_radius = st.number_input(
                "Probe Radius (Å)",
                min_value=0.1,
                value=2.0,
                step=0.1,
                help="Radio de la sonda para Alpha Shape"
            )
        
        with col2:
            smoothing = st.number_input(
                "Smoothing Iterations",
                min_value=0,
                value=3,
                step=1,
                help="Iteraciones de suavizado de malla"
            )
        
        with col3:
            num_ghost_layers = st.number_input(
                "Ghost Layers",
                min_value=1,
                value=3,
                step=1,
                help="Número de capas de ghost particles"
            )
        
        st.markdown("---")
        
        if st.button("🚀 Ejecutar Análisis Ghost Particles", type="primary", use_container_width=True):
            with st.spinner("⏳ Procesando..."):
                try:
                    # Extraer coordenadas
                    positions = df[['x', 'y', 'z']].values.astype(np.float32)
                    box_bounds = np.array(header['box_bounds'], dtype=np.float32)
                    
                    # Detectar parámetro de red automáticamente
                    if len(positions) > 1:
                        distances = np.linalg.norm(positions[0] - positions[1:], axis=1)
                        lattice_param = np.percentile(distances[distances > 0], 10)
                    else:
                        lattice_param = 1.0
                    
                    st.info(f"📏 Parámetro de red detectado: {lattice_param:.4f} Å")
                    
                    # Crear constructor con Ghost Particles - INTERFAZ CORRECTA
                    constructor = AlphaShapeWithGhosts(
                        positions=positions,
                        probe_radius=probe_radius,
                        box_bounds=box_bounds,
                        lattice_param=lattice_param,
                        num_ghost_layers=num_ghost_layers,
                        smoothing_level=smoothing
                    )
                    
                    # Ejecutar análisis
                    constructor.perform()
                    
                    # Obtener átomos de nanoporo
                    surface_atoms_indices = constructor.get_surface_atoms_indices()
                    
                    # Crear DataFrame filtrado
                    filtered_df = df.iloc[surface_atoms_indices].copy().reset_index(drop=True)
                    
                    # Guardar en sesión
                    st.session_state['alpha_result'] = {
                        'header': header,
                        'filtered_df': filtered_df,
                        'original_df': df,
                        'surface_atoms_indices': surface_atoms_indices,
                        'surface_area': getattr(constructor, 'surface_area', 0),
                        'n_vertices': len(getattr(constructor, 'surface_vertices', [])),
                        'n_faces': len(getattr(constructor, 'surface_faces', [])),
                        'n_ghosts': getattr(constructor, 'n_ghost_particles', 0),
                        'constructor': constructor
                    }
                    
                    st.success(f"✅ Análisis completado: **{len(surface_atoms_indices):,} átomos** detectados en nanoporos")
                    st.info(f"👻 Ghost particles: **{st.session_state['alpha_result']['n_ghosts']:,}** generadas")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    with st.expander("Detalles del error"):
                        st.code(traceback.format_exc())
        
        # RESULTADOS
        if st.session_state['alpha_result'] is not None:
            st.markdown("---")
            st.header("📈 Paso 3: Resultados del Análisis")
            
            result = st.session_state['alpha_result']
            filtered_df = result['filtered_df']
            original_df = result['original_df']
            
            # Métricas
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.metric("Nanopore Atoms", f"{len(filtered_df):,}")
            col2.metric("Total Atoms", f"{len(original_df):,}")
            
            percentage = 100 * len(filtered_df) / len(original_df) if len(original_df) > 0 else 0
            col3.metric("Percentage", f"{percentage:.2f}%")
            
            col4.metric("Surface Area", f"{result['surface_area']:.2f} Ų")
            col5.metric("Ghost Particles", f"{result['n_ghosts']:,}")
            
            st.markdown("---")
            
            # VISUALIZACIÓN 3D
            st.subheader("🎨 Visualización 3D Interactiva")
            
            col_viz1, col_viz2, col_viz3 = st.columns([1, 1, 2])
            
            with col_viz1:
                show_bulk = st.checkbox("Show Bulk Atoms", value=True)
            
            with col_viz2:
                show_surface = st.checkbox("Show Nanopore Atoms", value=True)
            
            with col_viz3:
                marker_size = st.slider("Marker Size", 1, 15, 5, key="marker_size")
            
            if show_bulk or show_surface:
                fig = create_3d_visualization(original_df, filtered_df, show_bulk, show_surface, marker_size)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Select at least one atom type to visualize")
            else:
                st.warning("⚠️ Select at least one atom type to visualize")
            
            # Distribución
            with st.expander("📊 Z-axis Distribution", expanded=False):
                dist_fig = create_distribution_plot(original_df, filtered_df)
                st.plotly_chart(dist_fig, use_container_width=True)
            
            st.markdown("---")
            
            # EXPORTACIÓN
            st.header("💾 Paso 4: Exportación de Resultados")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                output_filename = st.text_input(
                    "Output filename",
                    value=f"nanopores_filtered_{uploaded_file.name}",
                    help="LAMMPS dump file with filtered atoms"
                )
            
            with col2:
                st.write("")
                st.write("")
                export_button = st.button("📥 Export File", type="primary", use_container_width=True)
            
            if export_button:
                try:
                    temp_path = f"/tmp/{output_filename}"
                    write_lammps_dump_optimized(temp_path, result['header'], filtered_df)
                    
                    with open(temp_path, 'rb') as f:
                        file_bytes = f.read()
                    
                    st.download_button(
                        label="⬇️ Download Filtered File",
                        data=file_bytes,
                        file_name=output_filename,
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                    
                    st.success(f"✅ File ready: **{output_filename}**")
                    
                    with st.expander("📋 Final Summary", expanded=True):
                        st.markdown(f"""
                        ### Statistics:
                        - **Input atoms:** {len(original_df):,}
                        - **Nanopore atoms detected:** {len(filtered_df):,}
                        - **Removed atoms:** {len(original_df) - len(filtered_df):,} ({100*(len(original_df)-len(filtered_df))/len(original_df):.1f}%)
                        - **Surface area:** {result['surface_area']:.4f} Ų
                        - **Mesh faces:** {result['n_faces']:,}
                        - **Ghost particles:** {result['n_ghosts']:,}
                        - **Parameters:**
                          - Probe radius: {probe_radius} Å
                          - Smoothing: {smoothing} iterations
                          - Ghost layers: {num_ghost_layers}
                        """)
                    
                except Exception as e:
                    st.error(f"❌ Export error: {str(e)}")
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
    
    except Exception as e:
        st.error(f"❌ File processing error: {str(e)}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())

else:
    st.info("👆 **Load a LAMMPS dump file to start the analysis**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Features:
        - ✅ Automatic nanopore detection
        - ✅ Ghost particles method
        - ✅ Interactive 3D visualization
        - ✅ Surface area calculation
        - ✅ LAMMPS export format
        - ✅ Support for complex structures
        """)
    
    with col2:
        st.markdown("""
        ### 📄 Supported Formats:
        - **LAMMPS dump** (.dump, .txt)
        - **Required columns:** x, y, z
        - **Optional columns:** id, type, etc.
        
        ### 🔧 Requirements:
        - Python 3.8+
        - NumPy, SciPy, Pandas
        - Streamlit, Plotly
        """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🧬 Nanopore Detector | Ghost Particles Analysis | "
    "Optimized Performance"
    "</div>",
    unsafe_allow_html=True
)