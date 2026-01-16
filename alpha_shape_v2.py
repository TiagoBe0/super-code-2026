"""
Alpha Shape Surface Constructor - VERSIÓN MEJORADA
Implementa la técnica de "shrinking box" para evitar detección de átomos del borde

Inspirado en OVITO's ConstructSurfaceModifier approach
"""

import numpy as np
from scipy.spatial import Delaunay
import warnings


class AlphaShapeSurfaceConstructorV2:
    """
    Constructor de superficie usando Alpha Shape con pre-procesamiento para
    eliminar automáticamente partículas del borde de la caja de simulación.
    
    MEJORAS RESPECTO A VERSIÓN ORIGINAL:
    - Pre-procesamiento: shrink simulation box antes de Alpha Shape
    - Mapeo automático de índices clipeados → originales
    - Post-procesamiento opcional como red de seguridad
    - Mejor manejo de PBC (preparado para futuras extensiones)
    """
    
    def __init__(self, positions, probe_radius, box_bounds=None,
                 shrink_box=True, shrink_distance=None,
                 smoothing_level=0, pbc_flags=None, 
                 select_surface_particles=False):
        """
        Args:
            positions: Nx3 array de coordenadas atómicas
            probe_radius: radio de la esfera de prueba para filtrado
            box_bounds: límites de la caja ((xmin,xmax), (ymin,ymax), (zmin,zmax))
            shrink_box: si True, reduce la caja antes de Alpha Shape
            shrink_distance: cuánto reducir la caja (default: 2*probe_radius)
            smoothing_level: iteraciones de suavizado Laplaciano
            pbc_flags: tuple de 3 bools para PBC (x, y, z) [no implementado aún]
            select_surface_particles: marcar átomos en la superficie
        """
        self.positions_original = np.array(positions, dtype=np.float64)
        self.probe_radius = probe_radius
        self.box_bounds_original = box_bounds
        self.shrink_box = shrink_box
        self.shrink_distance = shrink_distance or (probe_radius * 2.0)
        self.smoothing_level = smoothing_level
        self.pbc_flags = pbc_flags or (False, False, False)
        self.select_surface_particles = select_surface_particles
        
        # Auto-detectar box bounds si no se proveen
        if self.box_bounds_original is None:
            self.box_bounds_original = self._compute_box_bounds(self.positions_original)
        
        # Datos procesados (después de shrinking)
        self.positions = None  # Posiciones después de pre-procesamiento
        self.box_bounds = None  # Box después de shrinking
        self._original_to_processed_map = None  # Índices originales de átomos procesados
        
        # Resultados de Alpha Shape
        self.surface_vertices = None
        self.surface_faces = None
        self.surface_particle_selection = None
        self.surface_area = None
        self._surface_atom_indices = None  # Índices en array procesado
        
    def _compute_box_bounds(self, positions):
        """Auto-detectar límites de la caja"""
        return tuple(
            (positions[:, i].min(), positions[:, i].max())
            for i in range(3)
        )
    
    def _preprocess_shrink_box(self):
        """
        PRE-PROCESAMIENTO: Reducir la caja de simulación
        
        Técnica inspirada en OVITO para evitar detección de superficies falsas
        en el borde de la caja.
        
        Returns:
            positions_clipped: posiciones dentro de la nueva caja
            clipped_indices: índices originales de las posiciones clipeadas
            new_box_bounds: nuevos límites de caja
        """
        print(f"\n[Pre-procesamiento] Reduciendo caja por {self.shrink_distance:.2f} Å en cada lado...")
        
        # Calcular nuevos límites
        new_box_bounds = tuple(
            (self.box_bounds_original[i][0] + self.shrink_distance,
             self.box_bounds_original[i][1] - self.shrink_distance)
            for i in range(3)
        )
        
        # Validar que la caja no se vuelva demasiado pequeña
        for i, (lo, hi) in enumerate(new_box_bounds):
            if hi <= lo:
                raise ValueError(
                    f"Shrink distance ({self.shrink_distance}) es demasiado grande "
                    f"para la dimensión {['X','Y','Z'][i]}. La caja resultante es inválida."
                )
        
        # Crear máscara para átomos dentro de la nueva caja
        mask = np.all([
            (self.positions_original[:, 0] >= new_box_bounds[0][0]) & 
            (self.positions_original[:, 0] <= new_box_bounds[0][1]),
            (self.positions_original[:, 1] >= new_box_bounds[1][0]) & 
            (self.positions_original[:, 1] <= new_box_bounds[1][1]),
            (self.positions_original[:, 2] >= new_box_bounds[2][0]) & 
            (self.positions_original[:, 2] <= new_box_bounds[2][1])
        ], axis=0)
        
        clipped_indices = np.where(mask)[0]
        positions_clipped = self.positions_original[mask]
        
        print(f"  ✓ Átomos removidos (borde): {len(self.positions_original) - len(positions_clipped)}")
        print(f"  ✓ Átomos restantes (interior): {len(positions_clipped)}")
        print(f"  ✓ Nueva caja: X=[{new_box_bounds[0][0]:.2f}, {new_box_bounds[0][1]:.2f}], "
              f"Y=[{new_box_bounds[1][0]:.2f}, {new_box_bounds[1][1]:.2f}], "
              f"Z=[{new_box_bounds[2][0]:.2f}, {new_box_bounds[2][1]:.2f}]")
        
        return positions_clipped, clipped_indices, new_box_bounds
    
    def perform(self):
        """
        Algoritmo principal: pre-procesamiento + Delaunay + filtrado alpha + construcción superficie
        """
        print(f"\n{'='*80}")
        print(f"Alpha Shape Surface Constructor V2 (con shrinking)")
        print(f"{'='*80}")
        
        if self.probe_radius <= 0:
            raise ValueError("Probe radius debe ser positivo")
        
        # PASO 0: Pre-procesamiento (Shrinking)
        if self.shrink_box:
            self.positions, self._original_to_processed_map, self.box_bounds = \
                self._preprocess_shrink_box()
            
            if len(self.positions) == 0:
                warnings.warn("No quedan átomos después del shrinking. "
                            "Reduce shrink_distance o desactiva shrink_box.")
                self.surface_vertices = np.array([])
                self.surface_faces = np.array([])
                self.surface_area = 0.0
                self._surface_atom_indices = np.array([], dtype=int)
                return self
        else:
            print("\n[Pre-procesamiento] Shrinking desactivado")
            self.positions = self.positions_original
            self.box_bounds = self.box_bounds_original
            self._original_to_processed_map = np.arange(len(self.positions_original))
        
        # PASO 1: Generar teselación de Delaunay
        print("\n[1/5] Generando teselación de Delaunay...")
        try:
            delaunay = Delaunay(self.positions)
            print(f"  ✓ Tetraedros totales: {len(delaunay.simplices)}")
        except Exception as e:
            raise RuntimeError(f"Falló la teselación de Delaunay: {e}")
        
        # PASO 2: Filtrar tetraedros por parámetro alpha
        print(f"\n[2/5] Filtrando tetraedros (alpha = {self.probe_radius**2:.2f})...")
        valid_tets = self._filter_tetrahedra(delaunay)
        print(f"  ✓ Tetraedros válidos: {len(valid_tets)} / {len(delaunay.simplices)} "
              f"({100*len(valid_tets)/len(delaunay.simplices):.1f}%)")
        
        # PASO 3: Extraer facetas de superficie
        print("\n[3/5] Extrayendo facetas de superficie...")
        surface_facets = self._extract_surface_facets(delaunay, valid_tets)
        print(f"  ✓ Facetas de superficie: {len(surface_facets)}")
        
        # PASO 4: Construir malla de superficie
        print("\n[4/5] Construyendo malla de superficie...")
        self.surface_vertices, self.surface_faces = self._build_mesh(
            delaunay, surface_facets, valid_tets
        )
        print(f"  ✓ Vértices: {len(self.surface_vertices)}")
        print(f"  ✓ Caras: {len(self.surface_faces)}")
        
        # PASO 5: Marcar partículas superficiales
        if self.select_surface_particles:
            self.surface_particle_selection = self._select_surface_particles(
                delaunay, surface_facets
            )
        
        # PASO 6: Suavizado de malla
        if self.smoothing_level > 0:
            print(f"\n[5/5] Suavizando malla ({self.smoothing_level} iteraciones)...")
            self.surface_vertices = self._smooth_mesh(
                self.surface_vertices, self.surface_faces, self.smoothing_level
            )
        
        # PASO 7: Calcular área de superficie
        self.surface_area = self._compute_surface_area()
        
        print(f"\n{'='*80}")
        print(f"✓ Completado")
        print(f"  Área de superficie: {self.surface_area:.4f}")
        print(f"  Átomos superficiales (procesados): {len(self._surface_atom_indices)}")
        print(f"{'='*80}\n")
        
        return self
    
    def _filter_tetrahedra(self, delaunay):
        """Filtrar tetraedros por circumradius <= probe_radius"""
        valid_tets = []
        for tet_idx, tet in enumerate(delaunay.simplices):
            verts = self.positions[tet]
            circumradius = self._compute_circumradius(verts)
            if circumradius <= self.probe_radius:
                valid_tets.append(tet_idx)
        return np.array(valid_tets)
    
    def _compute_circumradius(self, vertices):
        """Calcular circumradius de un tetraedro"""
        v0, v1, v2, v3 = vertices
        
        # Volumen usando producto triple escalar
        a = v1 - v0
        b = v2 - v0
        c = v3 - v0
        volume = abs(np.dot(a, np.cross(b, c))) / 6.0
        
        if volume < 1e-12:  # Tetraedro degenerado
            return np.inf
        
        # Resolver sistema lineal para encontrar centro de la circumesfera
        A = np.array([
            2*(v1 - v0),
            2*(v2 - v0),
            2*(v3 - v0)
        ])
        
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
        """
        Extraer facetas de superficie:
        Una faceta está en la superficie si limita con exactamente 1 tetraedro válido
        """
        valid_tet_set = set(valid_tets)
        facet_to_tets = {}
        
        for tet_idx, tet in enumerate(delaunay.simplices):
            is_valid = tet_idx in valid_tet_set
            
            # 4 facetas por tetraedro (opuesto a cada vértice)
            for i in range(4):
                facet = tuple(sorted(np.delete(tet, i)))
                facet_key = frozenset(facet)
                
                if facet_key not in facet_to_tets:
                    facet_to_tets[facet_key] = []
                facet_to_tets[facet_key].append((tet_idx, is_valid))
        
        # Facetas de superficie: limitan con exactamente 1 tet válido
        surface_facets = []
        for facet_key, tet_list in facet_to_tets.items():
            valid_count = sum(1 for _, is_valid in tet_list if is_valid)
            if valid_count == 1:
                surface_facets.append(list(facet_key))
        
        return surface_facets
    
    def _build_mesh(self, delaunay, surface_facets, valid_tets):
        """Construir malla de superficie desde facetas"""
        if not surface_facets:
            self._surface_atom_indices = np.array([], dtype=int)
            return np.array([]), np.array([])
        
        # Índices únicos de vértices en la superficie
        surface_vertex_indices = sorted(set(np.array(surface_facets).flatten()))
        
        # Guardar índices en el array PROCESADO
        self._surface_atom_indices = np.array(surface_vertex_indices, dtype=int)
        
        # Mapeo de índices antiguos → nuevos
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(surface_vertex_indices)}
        
        # Extraer vértices
        vertices = self.positions[surface_vertex_indices]
        
        # Remapear caras
        faces = []
        for facet in surface_facets:
            remapped_facet = [vertex_map[v] for v in facet]
            faces.append(remapped_facet)
        
        return vertices, np.array(faces)
    
    def _select_surface_particles(self, delaunay, surface_facets):
        """Marcar partículas de entrada que están en la superficie"""
        selection = np.zeros(len(self.positions), dtype=np.int32)
        for facet in surface_facets:
            for particle_idx in facet:
                selection[particle_idx] = 1
        return selection
    
    def _smooth_mesh(self, vertices, faces, iterations):
        """Suavizado Laplaciano de vértices de la malla"""
        smoothed_vertices = vertices.copy()
        
        for iteration in range(iterations):
            new_vertices = smoothed_vertices.copy()
            
            # Construir adyacencias
            vertex_neighbors = [set() for _ in range(len(smoothed_vertices))]
            for face in faces:
                for i in range(len(face)):
                    for j in range(len(face)):
                        if i != j:
                            vertex_neighbors[face[i]].add(face[j])
            
            # Actualizar cada vértice interior
            for v_idx in range(len(smoothed_vertices)):
                neighbors = list(vertex_neighbors[v_idx])
                if neighbors and len(neighbors) > 2:  # Vértice interior
                    new_vertices[v_idx] = smoothed_vertices[neighbors].mean(axis=0)
            
            smoothed_vertices = new_vertices
        
        return smoothed_vertices
    
    def _compute_surface_area(self):
        """Calcular área total de la superficie"""
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
    
    # ============================================================================
    # MÉTODOS PÚBLICOS PARA OBTENER RESULTADOS
    # ============================================================================
    
    def get_surface_atoms_original_indices(self):
        """
        Obtener índices de átomos superficiales en el array ORIGINAL
        (antes del pre-procesamiento)
        
        Returns:
            np.array: índices de átomos superficiales en positions_original
        """
        if self._surface_atom_indices is None or len(self._surface_atom_indices) == 0:
            return np.array([], dtype=int)
        
        # Mapear: índices procesados → índices originales
        return self._original_to_processed_map[self._surface_atom_indices]
    
    def get_surface_atoms_processed_indices(self):
        """
        Obtener índices de átomos superficiales en el array PROCESADO
        (después del pre-procesamiento)
        
        Returns:
            np.array: índices de átomos superficiales en positions (procesado)
        """
        return self._surface_atom_indices if self._surface_atom_indices is not None else np.array([], dtype=int)
    
    def export_filtered_atoms_dump(self, output_filename, atom_data=None):
        """
        Exportar átomos superficiales a archivo LAMMPS dump
        
        Args:
            output_filename: ruta del archivo de salida
            atom_data: dict con datos originales de átomos (id, type, x, y, z, etc.)
        """
        original_indices = self.get_surface_atoms_original_indices()
        
        if len(original_indices) == 0:
            print("⚠️ No hay átomos superficiales para exportar")
            return
        
        with open(output_filename, 'w') as f:
            f.write("ITEM: TIMESTEP\n")
            f.write("0\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{len(original_indices)}\n")
            
            # Box bounds (usar los originales)
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            for i in range(3):
                f.write(f"{self.box_bounds_original[i][0]:.6f} {self.box_bounds_original[i][1]:.6f}\n")
            
            # Datos de átomos
            if atom_data is not None:
                header_cols = atom_data.get('columns', ['id', 'type', 'x', 'y', 'z'])
                f.write(f"ITEM: ATOMS {' '.join(header_cols)}\n")
                
                for orig_idx in original_indices:
                    row_data = []
                    for col in header_cols:
                        if col == 'id':
                            row_data.append(str(orig_idx + 1))  # LAMMPS usa 1-based
                        elif col in atom_data:
                            row_data.append(str(atom_data[col][orig_idx]))
                        else:
                            row_data.append("0")
                    f.write(" ".join(row_data) + "\n")
            else:
                # Formato simple
                f.write("ITEM: ATOMS id type x y z\n")
                for orig_idx in original_indices:
                    atom_id = orig_idx + 1
                    atom_type = 1
                    x, y, z = self.positions_original[orig_idx]
                    f.write(f"{atom_id} {atom_type} {x:.8f} {y:.8f} {z:.8f}\n")
        
        print(f"✓ Exportados {len(original_indices)} átomos superficiales a {output_filename}")


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("Alpha Shape Surface Constructor V2 - Ejemplo de Uso\n")
    
    # Generar datos sintéticos
    np.random.seed(42)
    n_atoms = 100
    positions = np.random.rand(n_atoms, 3) * 20  # Caja de 20x20x20
    
    # Parámetros
    probe_radius = 3.0
    shrink_distance = 2.5  # Excluir 2.5 Å del borde
    
    # Ejecutar Alpha Shape con shrinking
    print("=" * 80)
    print("TEST: Alpha Shape con pre-procesamiento (shrinking)")
    print("=" * 80)
    
    constructor = AlphaShapeSurfaceConstructorV2(
        positions=positions,
        probe_radius=probe_radius,
        shrink_box=True,
        shrink_distance=shrink_distance,
        smoothing_level=2,
        select_surface_particles=True
    )
    
    constructor.perform()
    
    # Obtener resultados
    surface_atoms_original = constructor.get_surface_atoms_original_indices()
    
    print(f"\n{'='*80}")
    print(f"RESULTADOS FINALES")
    print(f"{'='*80}")
    print(f"Átomos totales (originales): {len(positions)}")
    print(f"Átomos procesados (después de shrinking): {len(constructor.positions)}")
    print(f"Átomos superficiales detectados: {len(surface_atoms_original)}")
    print(f"Índices de átomos superficiales (originales): {surface_atoms_original[:10]}...")
    print(f"Área de superficie: {constructor.surface_area:.4f}")
    print(f"Vértices de malla: {len(constructor.surface_vertices)}")
    print(f"Caras de malla: {len(constructor.surface_faces)}")
    print(f"{'='*80}\n")