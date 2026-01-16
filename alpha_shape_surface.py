import streamlit as st
import pandas as pd
import numpy as np
import traceback
from pathlib import Path
from io import StringIO
from typing import Dict, Tuple, Any

from scipy.spatial import Delaunay
import plotly.express as px

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Nanoporos - Alpha Shape",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Filtrado de Nanoporos con Alpha Shape")
st.markdown("**Detección de átomos en superficie de nanoporos usando triangulación de Delaunay**")
st.markdown("---")

# ==========================================
# PARSERS PARA LAMMPS DUMP
# ==========================================

@st.cache_data
def parse_lammps_dump(file_content: bytes) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Lee archivo LAMMPS dump con manejo robusto de errores"""
    lines = file_content.decode('utf-8').split('\n')
    
    header = {'box_bounds': []}
    atom_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line == "ITEM: TIMESTEP":
            header['timestep'] = int(lines[i+1].strip())
            i += 2
        elif line == "ITEM: NUMBER OF ATOMS":
            header['n_atoms_header'] = int(lines[i+1].strip())
            i += 2
        elif line.startswith("ITEM: BOX BOUNDS"):
            i += 1
            for _ in range(3):
                if i < len(lines):
                    bound_line = lines[i].strip()
                    bound_line = bound_line.replace('0.00.0', '0.0 0.0').replace('105.60.0', '105.6 0.0')
                    parts = [float(x) for x in bound_line.split()]
                    if parts:
                        lo, hi = min(parts), max(parts)
                        header['box_bounds'].append([lo, hi])
                    else:
                        header['box_bounds'].append([0.0, 0.0])
                    i += 1
        elif line.startswith("ITEM: ATOMS"):
            columns = line.split()[2:]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("ITEM:"):
                atom_line = lines[i].strip()
                parts = atom_line.split()
                if len(parts) == len(columns):
                    atom_lines.append(atom_line + '\n')
                i += 1
        else:
            i += 1
    
    if not atom_lines:
        raise ValueError("No se encontraron líneas de datos válidas")
    
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
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        for bounds in header['box_bounds']:
            f.write(f"{bounds[0]} {bounds[1]}\n")
        f.write(f"ITEM: ATOMS {' '.join(df.columns)}\n")
        f.write(df.to_string(header=False, index=False, float_format="%.8f"))
        f.write("\n")


# ==========================================
# MÓDULO ALPHA SHAPE
# ==========================================

class AlphaShapeSurfaceConstructor:
    """Construye superficie usando Alpha Shape"""
    
    def __init__(self, positions, probe_radius, smoothing_level=0):
        self.positions = np.array(positions, dtype=np.float64)
        self.probe_radius = probe_radius
        self.smoothing_level = smoothing_level
        self.surface_vertices = None
        self.surface_faces = None
        self._surface_atom_indices = None
        self.surface_area = None
    
    def perform(self):
        """Algoritmo principal"""
        if self.probe_radius <= 0:
            raise ValueError("Probe radius debe ser positivo")
        
        delaunay = Delaunay(self.positions)
        valid_tets = self._filter_tetrahedra(delaunay)
        surface_facets = self._extract_surface_facets(delaunay, valid_tets)
        self.surface_vertices, self.surface_faces = self._build_mesh(delaunay, surface_facets)
        self.surface_area = self._compute_surface_area()
        
        return self
    
    def _filter_tetrahedra(self, delaunay):
        """Filtra tetraedros por circumradius"""
        valid_tets = []
        for tet_idx, tet in enumerate(delaunay.simplices):
            verts = self.positions[tet]
            circumradius = self._compute_circumradius(verts)
            if circumradius <= self.probe_radius:
                valid_tets.append(tet_idx)
        return np.array(valid_tets)
    
    def _compute_circumradius(self, vertices):
        """Calcula circumradius de tetraedro"""
        v0, v1, v2, v3 = vertices
        a = v1 - v0
        b = v2 - v0
        c = v3 - v0
        volume = abs(np.dot(a, np.cross(b, c))) / 6.0
        
        if volume < 1e-12:
            return np.inf
        
        A = np.array([2*(v1 - v0), 2*(v2 - v0), 2*(v3 - v0)])
        b_vec = np.array([
            np.dot(v1, v1) - np.dot(v0, v0),
            np.dot(v2, v2) - np.dot(v0, v0),
            np.dot(v3, v3) - np.dot(v0, v0)
        ])
        
        try:
            center = np.linalg.solve(A, b_vec)
            R = np.linalg.norm(center - v0)
            return R
        except np.linalg.LinAlgError:
            return np.inf
    
    def _extract_surface_facets(self, delaunay, valid_tets):
        """Extrae facetas de superficie"""
        valid_tet_set = set(valid_tets)
        facet_to_tets = {}
        
        for tet_idx, tet in enumerate(delaunay.simplices):
            is_valid = tet_idx in valid_tet_set
            for i in range(4):
                facet = tuple(sorted(np.delete(tet, i)))
                facet_key = frozenset(facet)
                if facet_key not in facet_to_tets:
                    facet_to_tets[facet_key] = []
                facet_to_tets[facet_key].append((tet_idx, is_valid))
        
        surface_facets = []
        for facet_key, tet_list in facet_to_tets.items():
            valid_count = sum(1 for _, is_valid in tet_list if is_valid)
            if valid_count == 1:
                surface_facets.append(list(facet_key))
        
        return surface_facets
    
    def _build_mesh(self, delaunay, surface_facets):
        """Construye malla de superficie"""
        if not surface_facets:
            self._surface_atom_indices = np.array([], dtype=int)
            return np.array([]), np.array([])
        
        surface_vertex_indices = sorted(set(np.array(surface_facets).flatten()))
        self._surface_atom_indices = np.array(surface_vertex_indices, dtype=int)
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(surface_vertex_indices)}
        
        vertices = self.positions[surface_vertex_indices]
        faces = [[vertex_map[v] for v in facet] for facet in surface_facets]
        
        return vertices, np.array(faces)
    
    def _compute_surface_area(self):
        """Calcula área de superficie"""
        if self.surface_faces is None or len(self.surface_faces) == 0:
            return 0.0
        
        total_area = 0.0
        for face in self.surface_faces:
            if len(face) == 3:
                v0, v1, v2 = self.surface_vertices[face]
                edge1 = v1 - v0
                edge2 = v2 - v0
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                total_area += area
        
        return total_area
    
    def filter_surface_atoms_exclude_box_boundary(self, box_bounds=None, margin=0.01):
        """Filtra átomos de superficie excluyendo borde"""
        if len(self._surface_atom_indices) == 0:
            return np.array([], dtype=int)
        
        if box_bounds is None:
            box_bounds = (
                (self.positions[:, 0].min(), self.positions[:, 0].max()),
                (self.positions[:, 1].min(), self.positions[:, 1].max()),
                (self.positions[:, 2].min(), self.positions[:, 2].max())
            )
        
        interior_atoms = []
        for atom_idx in self._surface_atom_indices:
            pos = self.positions[atom_idx]
            on_boundary = False
            
            for dim in range(3):
                dist_to_min = pos[dim] - box_bounds[dim][0]
                dist_to_max = box_bounds[dim][1] - pos[dim]
                if dist_to_min < margin or dist_to_max < margin:
                    on_boundary = True
                    break
            
            if not on_boundary:
                interior_atoms.append(atom_idx)
        
        return np.array(interior_atoms, dtype=int)


# ==========================================
# INTERFAZ STREAMLIT
# ==========================================

st.header("📁 Paso 1: Cargar Archivo")
uploaded_file = st.file_uploader("Selecciona archivo LAMMPS .dump", type=['dump', 'txt'])

if uploaded_file:
    try:
        header, df = parse_lammps_dump(uploaded_file.getvalue())
        
        # Validar que existan las columnas necesarias
        required_cols = ['x', 'y', 'z']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ El archivo no contiene las columnas necesarias: {missing_cols}")
            st.info(f"Columnas disponibles: {list(df.columns)}")
        else:
            st.success(f"✓ Archivo cargado: {len(df)} átomos")
        
        # Información del archivo
        with st.expander("📊 Información del Archivo"):
            col1, col2 = st.columns(2)
            col1.metric("Total de átomos", len(df))
            col1.metric("Timestep", header['timestep'])
            
            col2.write("**Box Bounds:**")
            for i, (lo, hi) in enumerate(header['box_bounds']):
                col2.write(f"Dimensión {['X', 'Y', 'Z'][i]}: [{lo:.2f}, {hi:.2f}]")
        
        st.markdown("---")
        
        # PASO 2: CONFIGURACIÓN Y EJECUCIÓN
        st.header("⚙️ Paso 2: Configuración de Alpha Shape")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            probe_radius = st.number_input("Radio de sonda (Å)", value=2.0, step=0.5, 
                                          help="Radio máximo del circumradio de los tetraedros válidos")
        with col2:
            smoothing = st.number_input("Nivel de suavizado", value=1, step=1,
                                       help="Nivel de suavizado (no implementado)")
        with col3:
            boundary_margin = st.number_input("Margen de borde (Å)", value=0.1, step=0.05,
                                             help="Distancia mínima al borde de la caja de simulación")
        
        if st.button("🚀 Ejecutar Alpha Shape", type="primary"):
            with st.spinner("Procesando Alpha Shape..."):
                try:
                    positions = df[['x', 'y', 'z']].values
                    box_bounds = tuple(
                        (header['box_bounds'][i][0], header['box_bounds'][i][1])
                        for i in range(3)
                    )
                    
                    constructor = AlphaShapeSurfaceConstructor(
                        positions=positions,
                        probe_radius=probe_radius,
                        smoothing_level=smoothing
                    )
                    constructor.perform()
                    
                    surface_atoms = constructor.filter_surface_atoms_exclude_box_boundary(
                        box_bounds=box_bounds,
                        margin=boundary_margin
                    )
                    
                    filtered_df = df.iloc[surface_atoms].copy()
                    
                    st.session_state['alpha_result'] = {
                        'header': header,
                        'filtered_df': filtered_df,
                        'original_df': df,
                        'surface_atoms': surface_atoms,
                        'surface_area': constructor.surface_area
                    }
                    
                    st.success(f"✓ Alpha Shape completado: {len(surface_atoms)} átomos detectados")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.code(traceback.format_exc())
        
        # MOSTRAR RESULTADOS
        if 'alpha_result' in st.session_state:
            st.markdown("---")
            st.header("📈 Paso 3: Resultados")
            
            result = st.session_state['alpha_result']
            filtered_df = result['filtered_df']
            original_df = result['original_df']
            
            # Métricas
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Átomos detectados", len(filtered_df))
            col2.metric("Átomos originales", len(original_df))
            col3.metric("Porcentaje", f"{100*len(filtered_df)/len(original_df):.1f}%")
            col4.metric("Área superficie", f"{result['surface_area']:.2f} Ų")
            
            # VISUALIZACIÓN 3D
            with st.expander("🎨 Visualización 3D - Nanoporos Detectados", expanded=True):
                df_all_atoms = original_df.copy()
                df_all_atoms['Tipo'] = 'Bulk'
                
                df_filtered_copy = filtered_df.copy()
                df_filtered_copy['Tipo'] = 'Nanoporo'
                
                df_combined = pd.concat([df_all_atoms, df_filtered_copy], ignore_index=True)
                
                fig = px.scatter_3d(
                    df_combined,
                    x='x', y='y', z='z',
                    color='Tipo',
                    color_discrete_map={'Bulk': 'lightblue', 'Nanoporo': 'red'},
                    title='Átomos de nanoporos (rojo) vs Bulk (azul)',
                    labels={'x': 'X (Å)', 'y': 'Y (Å)', 'z': 'Z (Å)'},
                )
                
                fig.update_traces(marker=dict(size=4))
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
            
            # EXPORTACIÓN
            st.markdown("---")
            st.header("💾 Paso 4: Exportación")
            
            col1, col2 = st.columns(2)
            with col1:
                output_filename = st.text_input("Nombre del archivo", value="nanopores_filtered.dump")
            with col2:
                st.write("")  # Espacio
            
            if st.button("📥 Exportar Archivo Filtrado"):
                try:
                    output_path = Path(output_filename)
                    write_lammps_dump(str(output_path), result['header'], filtered_df)
                    
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Descargar Archivo",
                            data=f,
                            file_name=output_filename,
                            mime="application/octet-stream"
                        )
                    
                    st.success(f"✓ Archivo exportado: {output_filename}")
                    
                except Exception as e:
                    st.error(f"❌ Error al exportar: {e}")
        
    except Exception as e:
        st.error(f"❌ Error al leer archivo: {e}")
        st.code(traceback.format_exc())

else:
    st.info("👆 Carga un archivo .dump para comenzar el análisis")
    
    # Instrucciones
    with st.expander("ℹ️ Instrucciones de Uso"):
        st.markdown("""
        ### Cómo usar esta herramienta:
        
        1. **Cargar Archivo**: Sube tu archivo LAMMPS dump (.dump o .txt)
        2. **Configurar Parámetros**:
           - **Radio de sonda**: Controla el tamaño máximo de los poros detectados
           - **Margen de borde**: Excluye átomos cerca del borde de la simulación
        3. **Ejecutar**: Haz clic en "Ejecutar Alpha Shape"
        4. **Visualizar**: Revisa los resultados en el gráfico 3D
        5. **Exportar**: Descarga el archivo filtrado con solo los átomos de nanoporos
        
        ### Algoritmo Alpha Shape:
        - Usa triangulación de Delaunay para construir una malla 3D
        - Filtra tetraedros por su circumradio
        - Identifica átomos en la superficie de nanoporos
        - Excluye átomos en los bordes de la caja de simulación
        """)