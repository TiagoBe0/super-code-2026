"""
🧬 NANOPORE DETECTOR - Alpha Shape Surface Constructor
Interfaz visual para detección de átomos en superficies de nanoporos
usando el algoritmo Alpha Shape (Delaunay + filtrado por circumradius)
"""

import streamlit as st
import pandas as pd
import numpy as np
import traceback
from pathlib import Path
from io import StringIO, BytesIO
from typing import Dict, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px

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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧬 Nanopore Detector - Alpha Shape Analysis</div>', unsafe_allow_html=True)
st.markdown("**Identificación automática de átomos en superficies de nanoporos mediante Alpha Shape**")
st.markdown("---")

# ==========================================
# IMPORTAR MÓDULOS ALPHA SHAPE
# ==========================================

ALPHA_SHAPE_BASIC_AVAILABLE = False
ALPHA_SHAPE_GHOST_AVAILABLE = False

try:
    import sys
    sys.path.insert(0, '/mnt/project')
    sys.path.insert(0, '/mnt/user-data/uploads')
    sys.path.insert(0, '.')
    
    # Intentar importar versión con Ghost Particles (preferida)
    try:
        from alpha_shape_ghost_particles import AlphaShapeWithGhosts
        ALPHA_SHAPE_GHOST_AVAILABLE = True
    except ImportError:
        pass
    
    # Intentar importar versión básica (fallback)
    try:
        from alpha_shape_surface import AlphaShapeSurfaceConstructor
        ALPHA_SHAPE_BASIC_AVAILABLE = True
    except ImportError:
        pass
    
    if not ALPHA_SHAPE_GHOST_AVAILABLE and not ALPHA_SHAPE_BASIC_AVAILABLE:
        st.error("⚠️ No se pudo importar ningún módulo Alpha Shape. Asegúrate que alpha_shape_ghost_particles.py o alpha_shape_surface.py estén disponibles.")
        
except Exception as e:
    st.error(f"❌ Error al importar módulos: {e}")

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
                    # Limpiar posibles errores de formato
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
        
        # Escribir datos átomo por átomo
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
# FUNCIONES DE VISUALIZACIÓN
# ==========================================

def create_3d_visualization(original_df, filtered_df, show_bulk, show_surface, marker_size):
    """Crea visualización 3D interactiva con Plotly"""
    
    df_to_plot = []
    
    if show_bulk:
        df_bulk = original_df.copy()
        df_bulk['Tipo'] = 'Bulk'
        df_bulk['Color'] = 'lightblue'
        df_to_plot.append(df_bulk)
    
    if show_surface:
        df_surf = filtered_df.copy()
        df_surf['Tipo'] = 'Nanoporo'
        df_surf['Color'] = 'red'
        df_to_plot.append(df_surf)
    
    if not df_to_plot:
        return None
    
    df_combined = pd.concat(df_to_plot, ignore_index=True)
    
    # Crear figura con Plotly
    fig = px.scatter_3d(
        df_combined,
        x='x', y='y', z='z',
        color='Tipo',
        color_discrete_map={'Bulk': '#87CEEB', 'Nanoporo': '#FF4444'},
        title='Visualización 3D: Átomos de Nanoporos (rojo) vs Bulk (azul)',
        labels={'x': 'X (Å)', 'y': 'Y (Å)', 'z': 'Z (Å)'},
        height=700
    )
    
    fig.update_traces(
        marker=dict(
            size=marker_size,
            opacity=0.8 if show_bulk else 1.0,
            line=dict(width=0)
        )
    )
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            yaxis=dict(backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            zaxis=dict(backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_distribution_plot(original_df, filtered_df):
    """Crea gráfico de distribución de átomos"""
    fig = go.Figure()
    
    # Histograma de átomos originales
    fig.add_trace(go.Histogram(
        x=original_df['z'],
        name='Todos los átomos',
        opacity=0.6,
        marker_color='lightblue',
        nbinsx=50
    ))
    
    # Histograma de átomos de nanoporos
    fig.add_trace(go.Histogram(
        x=filtered_df['z'],
        name='Átomos de nanoporos',
        opacity=0.8,
        marker_color='red',
        nbinsx=50
    ))
    
    fig.update_layout(
        title='Distribución de Átomos en Eje Z',
        xaxis_title='Z (Å)',
        yaxis_title='Frecuencia',
        barmode='overlay',
        height=400
    )
    
    return fig


# ==========================================
# BARRA LATERAL - INFORMACIÓN
# ==========================================

with st.sidebar:
    st.header("ℹ️ Información")
    
    with st.expander("📖 Cómo usar", expanded=True):
        st.markdown("""
        ### Pasos:
        1. **Cargar archivo** LAMMPS dump
        2. **Configurar parámetros** del análisis
        3. **Ejecutar** Alpha Shape
        4. **Visualizar** resultados en 3D
        5. **Exportar** archivo filtrado
        
        ### Parámetros clave:
        - **Radio de sonda**: Tamaño máximo de poros detectados
        - **Suavizado**: Iteraciones de suavizado Laplaciano
        - **Margen de borde**: Exclusión de átomos del borde
        """)
    
    with st.expander("🔬 Algoritmo Alpha Shape"):
        st.markdown("""
        **Método:**
        1. Teselación de Delaunay 3D
        2. Filtrado por circumradius ≤ probe_radius
        3. Extracción de facetas superficiales
        4. Construcción de malla triangular
        5. Post-procesamiento (exclusión de bordes)
        
        **Ventajas:**
        - Detección precisa de cavidades
        - Robusto para geometrías complejas
        - Compatible con OVITO
        """)
    
    with st.expander("💡 Tips"):
        st.markdown("""
        - Radio de sonda típico: 1.5-3.0 Å
        - Mayor suavizado = superficie más regular
        - Margen de borde ≥ 0.1 Å recomendado
        - Para FCC/BCC: radio ~ 1.8-2.2 Å
        """)
    
    st.markdown("---")
    st.markdown("**Desarrollado con:** Python + Streamlit")
    st.markdown("**Algoritmo:** Alpha Shape (SciPy)")

# ==========================================
# ÁREA PRINCIPAL
# ==========================================

# PASO 1: CARGA DE ARCHIVO
st.header("📁 Paso 1: Cargar Archivo")

uploaded_file = st.file_uploader(
    "Selecciona tu archivo LAMMPS dump",
    type=['dump', 'txt', 'data'],
    help="Archivo en formato LAMMPS dump con coordenadas atómicas"
)

if uploaded_file:
    try:
        with st.spinner("Leyendo archivo..."):
            header, df = parse_lammps_dump(uploaded_file.getvalue())
        
        # Validar columnas necesarias
        required_cols = ['x', 'y', 'z']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Faltan columnas necesarias: {missing_cols}")
            st.info(f"Columnas disponibles: {list(df.columns)}")
            st.stop()
        
        st.success(f"✅ Archivo cargado exitosamente: **{len(df)} átomos**")
        
        # Guardar en session_state
        st.session_state['uploaded_data'] = {
            'header': header,
            'df': df,
            'filename': uploaded_file.name
        }
        
        # Mostrar información del archivo
        with st.expander("📊 Información del Archivo", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total de átomos", f"{len(df):,}")
                st.metric("Timestep", header['timestep'])
            
            with col2:
                st.write("**Box Bounds:**")
                for i, (lo, hi) in enumerate(header['box_bounds']):
                    st.write(f"• {['X', 'Y', 'Z'][i]}: [{lo:.2f}, {hi:.2f}] Å")
            
            with col3:
                st.write("**Columnas disponibles:**")
                for col in df.columns:
                    st.write(f"• {col}")
        
        st.markdown("---")
        
        # PASO 2: CONFIGURACIÓN
        st.header("⚙️ Paso 2: Configuración de Parámetros")
        
        # Selector de método
        st.subheader("🔬 Método de Análisis")
        
        method_options = []
        if ALPHA_SHAPE_GHOST_AVAILABLE:
            method_options.append("Ghost Particles (Recomendado)")
        if ALPHA_SHAPE_BASIC_AVAILABLE:
            method_options.append("Alpha Shape Básico")
        
        if not method_options:
            st.error("❌ No hay métodos disponibles. Verifica los archivos de módulos.")
            st.stop()
        
        selected_method = st.selectbox(
            "Selecciona el método de análisis",
            options=method_options,
            help="Ghost Particles: Replica átomos en bordes para evitar superficies falsas (más preciso)\nAlpha Shape Básico: Método directo sin replicación"
        )
        
        use_ghost_particles = "Ghost" in selected_method
        
        if use_ghost_particles:
            st.info("✨ **Ghost Particles**: Este método replica átomos en los bordes de la caja para eliminar artefactos de superficie artificial en las fronteras.")
        
        st.markdown("---")
        st.subheader("⚙️ Parámetros del Algoritmo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            probe_radius = st.number_input(
                "Radio de sonda (Å)",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Radio máximo del circumradio de tetraedros válidos"
            )
        
        with col2:
            smoothing = st.number_input(
                "Nivel de suavizado",
                min_value=0,
                max_value=20,
                value=1,
                step=1,
                help="Iteraciones de suavizado Laplaciano (0 = sin suavizado)"
            )
        
        with col3:
            boundary_margin = st.number_input(
                "Margen de borde (Å)",
                min_value=0.0,
                max_value=5.0,
                value=0.1,
                step=0.05,
                help="Distancia mínima al borde de la caja"
            )
        
        with col4:
            exclude_boundary = st.checkbox(
                "Excluir bordes",
                value=True,
                help="Eliminar átomos en bordes de la caja"
            )
        
        # Parámetros adicionales para Ghost Particles
        if use_ghost_particles and ALPHA_SHAPE_GHOST_AVAILABLE:
            st.markdown("---")
            st.subheader("👻 Parámetros de Ghost Particles")
            
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                lattice_param = st.number_input(
                    "Parámetro de red (Å)",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.0,
                    step=0.1,
                    help="Parámetro de red cristalino. 0 = auto-detectar. Típico FCC: ~4.0 Å"
                )
                lattice_param = None if lattice_param == 0.0 else lattice_param
            
            with col_g2:
                num_ghost_layers = st.number_input(
                    "Capas fantasma",
                    min_value=1,
                    max_value=5,
                    value=2,
                    step=1,
                    help="Número de capas atómicas a replicar en cada borde"
                )
        
        st.markdown("---")
        
        # Botón de ejecución
        col_button, col_info = st.columns([1, 3])
        
        with col_button:
            run_button = st.button(
                "🚀 Ejecutar Alpha Shape",
                type="primary",
                use_container_width=True
            )
        
        with col_info:
            if run_button:
                st.info("⏳ Procesando... Esto puede tardar unos momentos para sistemas grandes")
        
        # PASO 3: EJECUCIÓN
        if run_button:
            if not ALPHA_SHAPE_GHOST_AVAILABLE and not ALPHA_SHAPE_BASIC_AVAILABLE:
                st.error("❌ No hay módulos Alpha Shape disponibles")
                st.stop()
            
            with st.spinner("🔄 Ejecutando algoritmo Alpha Shape..."):
                try:
                    # Extraer posiciones
                    positions = df[['x', 'y', 'z']].values
                    box_bounds = tuple(
                        (header['box_bounds'][i][0], header['box_bounds'][i][1])
                        for i in range(3)
                    )
                    
                    # Crear constructor según el método seleccionado
                    if use_ghost_particles and ALPHA_SHAPE_GHOST_AVAILABLE:
                        st.info("🔄 Usando método Ghost Particles (avanzado)...")
                        constructor = AlphaShapeWithGhosts(
                            positions=positions,
                            probe_radius=probe_radius,
                            box_bounds=box_bounds,
                            lattice_param=lattice_param,
                            num_ghost_layers=num_ghost_layers,
                            smoothing_level=smoothing
                        )
                    else:
                        st.info("🔄 Usando método Alpha Shape básico...")
                        constructor = AlphaShapeSurfaceConstructor(
                            positions=positions,
                            probe_radius=probe_radius,
                            smoothing_level=smoothing,
                            select_surface_particles=True
                        )
                    
                    # Ejecutar
                    constructor.perform()
                    
                    # Filtrar átomos según el método
                    if use_ghost_particles and ALPHA_SHAPE_GHOST_AVAILABLE:
                        # Para Ghost Particles, los índices ya están filtrados automáticamente
                        surface_atoms_indices = constructor.get_surface_atoms_indices()
                    else:
                        # Para método básico, aplicar filtrado de bordes si está habilitado
                        if exclude_boundary:
                            surface_atoms_indices = constructor.get_filtered_surface_atoms(
                                exclude_box_boundary=True,
                                box_bounds=box_bounds,
                                margin=boundary_margin
                            )
                        else:
                            surface_atoms_indices = constructor._surface_atom_indices
                    
                    # Crear DataFrame filtrado
                    filtered_df = df.iloc[surface_atoms_indices].copy().reset_index(drop=True)
                    
                    # Guardar resultados
                    st.session_state['alpha_result'] = {
                        'header': header,
                        'filtered_df': filtered_df,
                        'original_df': df,
                        'surface_atoms_indices': surface_atoms_indices,
                        'surface_area': constructor.surface_area,
                        'n_vertices': len(constructor.surface_vertices),
                        'n_faces': len(constructor.surface_faces),
                        'constructor': constructor,
                        'method': 'Ghost Particles' if use_ghost_particles else 'Alpha Shape Básico'
                    }
                    
                    st.success(f"✅ Análisis completado: **{len(surface_atoms_indices)} átomos** detectados en nanoporos")
                    
                    # Mostrar info de ghost particles si está disponible
                    if use_ghost_particles and ALPHA_SHAPE_GHOST_AVAILABLE:
                        n_ghosts = getattr(constructor, 'n_ghost_particles', 0)
                        if n_ghosts > 0:
                            st.info(f"👻 Ghost particles generadas: **{n_ghosts:,}** ({n_ghosts/len(positions)*100:.1f}% del sistema original)")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error durante la ejecución: {str(e)}")
                    with st.expander("Ver detalles del error"):
                        st.code(traceback.format_exc())
        
        # PASO 4: RESULTADOS
        if 'alpha_result' in st.session_state:
            st.markdown("---")
            st.header("📈 Paso 3: Resultados del Análisis")
            
            result = st.session_state['alpha_result']
            filtered_df = result['filtered_df']
            original_df = result['original_df']
            
            # Mostrar método utilizado
            method_badge = "👻 Ghost Particles" if result.get('method') == 'Ghost Particles' else "🔬 Alpha Shape Básico"
            st.markdown(f"**Método utilizado:** {method_badge}")
            st.markdown("---")
            
            # Métricas principales
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.metric(
                "Átomos de nanoporos",
                f"{len(filtered_df):,}",
                delta=None
            )
            
            col2.metric(
                "Átomos totales",
                f"{len(original_df):,}",
                delta=None
            )
            
            percentage = 100 * len(filtered_df) / len(original_df) if len(original_df) > 0 else 0
            col3.metric(
                "Porcentaje",
                f"{percentage:.2f}%",
                delta=None
            )
            
            col4.metric(
                "Área superficie",
                f"{result['surface_area']:.2f} Ų",
                delta=None
            )
            
            col5.metric(
                "Vértices malla",
                f"{result['n_vertices']:,}",
                delta=None
            )
            
            st.markdown("---")
            
            # VISUALIZACIÓN 3D
            st.subheader("🎨 Visualización 3D Interactiva")
            
            col_viz1, col_viz2, col_viz3 = st.columns([1, 1, 2])
            
            with col_viz1:
                show_bulk = st.checkbox("Mostrar átomos Bulk", value=True)
            
            with col_viz2:
                show_surface = st.checkbox("Mostrar átomos Nanoporos", value=True)
            
            with col_viz3:
                marker_size = st.slider("Tamaño de marcadores", 1, 15, 5, key="marker_size")
            
            # Crear visualización
            fig = create_3d_visualization(original_df, filtered_df, show_bulk, show_surface, marker_size)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Selecciona al menos un tipo de átomo para visualizar")
            
            # Gráfico de distribución
            with st.expander("📊 Distribución de Átomos (Eje Z)", expanded=False):
                dist_fig = create_distribution_plot(original_df, filtered_df)
                st.plotly_chart(dist_fig, use_container_width=True)
            
            st.markdown("---")
            
            # PASO 5: EXPORTACIÓN
            st.header("💾 Paso 4: Exportación de Resultados")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                output_filename = st.text_input(
                    "Nombre del archivo de salida",
                    value=f"nanopores_filtered_{uploaded_file.name}",
                    help="Archivo LAMMPS dump con átomos filtrados"
                )
            
            with col2:
                st.write("")  # Espacio
                st.write("")  # Espacio
                export_button = st.button("📥 Exportar Archivo", type="primary", use_container_width=True)
            
            if export_button:
                try:
                    # Crear archivo temporal
                    temp_path = f"/tmp/{output_filename}"
                    write_lammps_dump(temp_path, result['header'], filtered_df)
                    
                    # Leer y preparar descarga
                    with open(temp_path, 'rb') as f:
                        file_bytes = f.read()
                    
                    st.download_button(
                        label="⬇️ Descargar Archivo Filtrado",
                        data=file_bytes,
                        file_name=output_filename,
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                    
                    st.success(f"✅ Archivo preparado: **{output_filename}**")
                    
                    # Resumen final
                    with st.expander("📋 Resumen del Proceso", expanded=True):
                        st.markdown(f"""
                        ### Estadísticas Finales:
                        - **Átomos de entrada:** {len(original_df):,}
                        - **Átomos detectados (nanoporos):** {len(filtered_df):,}
                        - **Átomos eliminados:** {len(original_df) - len(filtered_df):,} ({100*(len(original_df)-len(filtered_df))/len(original_df):.1f}%)
                        - **Área de superficie:** {result['surface_area']:.4f} Ų
                        - **Caras de malla:** {result['n_faces']:,}
                        - **Parámetros utilizados:**
                          - Radio de sonda: {probe_radius} Å
                          - Suavizado: {smoothing} iteraciones
                          - Margen de borde: {boundary_margin} Å
                        """)
                
                except Exception as e:
                    st.error(f"❌ Error al exportar: {str(e)}")
                    with st.expander("Ver detalles del error"):
                        st.code(traceback.format_exc())
    
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {str(e)}")
        with st.expander("Ver detalles del error"):
            st.code(traceback.format_exc())

else:
    # Pantalla inicial sin archivo
    st.info("👆 **Carga un archivo LAMMPS dump para comenzar el análisis**")
    
    # Información adicional
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Funcionalidades:
        - ✅ Detección automática de nanoporos
        - ✅ Visualización 3D interactiva
        - ✅ Filtrado de bordes de caja
        - ✅ Cálculo de área superficial
        - ✅ Exportación a formato LAMMPS
        - ✅ Compatible con FCC, BCC y estructuras complejas
        """)
    
    with col2:
        st.markdown("""
        ### 📄 Formatos soportados:
        - **LAMMPS dump** (.dump, .txt)
        - **Columnas requeridas:** x, y, z
        - **Columnas opcionales:** id, type, etc.
        
        ### 🔧 Requisitos:
        - Python 3.7+
        - NumPy, SciPy, Pandas
        - Streamlit, Plotly
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🧬 Nanopore Detector | Alpha Shape Surface Constructor | "
    "Powered by Python + Streamlit"
    "</div>",
    unsafe_allow_html=True
)