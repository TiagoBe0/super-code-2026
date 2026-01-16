"""
Alpha Shape con Ghost Particles (Técnica de OVITO)
Replica capas de átomos en los bordes para evitar detección de superficie falsa
"""

import numpy as np
from scipy.spatial import Delaunay
from collections import OrderedDict
import argparse


# ============================================================================
# DETECTOR AUTOMÁTICO DE PARÁMETRO DE RED
# ============================================================================

def detect_lattice_parameter(positions, sample_size=1000):
    """
    Detecta el parámetro de red analizando distancias entre vecinos cercanos
    
    Args:
        positions: array Nx3 de posiciones
        sample_size: número de átomos a muestrear
    
    Returns:
        lattice_param: parámetro de red detectado
    """
    from scipy.spatial import cKDTree
    
    # Muestrear átomos para acelerar
    if len(positions) > sample_size:
        indices = np.random.choice(len(positions), sample_size, replace=False)
        sample = positions[indices]
    else:
        sample = positions
    
    # Construir KD-Tree
    tree = cKDTree(sample)
    
    # Para cada átomo, encontrar vecinos cercanos
    distances_list = []
    for pos in sample[:min(100, len(sample))]:
        dists, _ = tree.query(pos, k=13)  # 12 vecinos + el mismo
        # Excluir distancia 0 (el mismo átomo)
        dists = dists[dists > 0.1]
        if len(dists) > 0:
            distances_list.extend(dists[:4])  # Primeros 4 vecinos
    
    if not distances_list:
        raise ValueError("No se pudieron encontrar vecinos")
    
    # El parámetro de red es la distancia más común entre primeros vecinos
    distances = np.array(distances_list)
    hist, bin_edges = np.histogram(distances, bins=50)
    peak_idx = np.argmax(hist)
    lattice_param = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
    
    print(f"\n[Auto-detección] Parámetro de red detectado: {lattice_param:.4f} Å")
    
    return lattice_param


# ============================================================================
# GENERADOR DE GHOST PARTICLES
# ============================================================================

def create_ghost_layers(positions, box_bounds, lattice_param, num_layers=2):
    """
    Crea capas fantasma en CARAS, ARISTAS y ESQUINAS del box (PBC completo)
    
    Args:
        positions: array Nx3 de posiciones originales
        box_bounds: ((xmin,xmax), (ymin,ymax), (zmin,zmax))
        lattice_param: parámetro de red cristalino
        num_layers: número de capas a replicar en cada cara
    
    Returns:
        ghost_positions: array con posiciones originales + fantasmas
        ghost_map: mapeo índice_ghost -> índice_original (-1 para originales)
        is_ghost: array booleano indicando si cada átomo es fantasma
    """
    
    print(f"\n{'='*80}")
    print(f"CREANDO GHOST PARTICLES COMPLETO (Caras + Aristas + Esquinas)")
    print(f"{'='*80}")
    print(f"\nParámetros:")
    print(f"  - Parámetro de red: {lattice_param:.4f} Å")
    print(f"  - Capas por cara: {num_layers}")
    print(f"  - Grosor total: {num_layers * lattice_param:.4f} Å")
    
    all_positions = list(positions)
    ghost_map = [-1] * len(positions)  # -1 = átomo original
    is_ghost = [False] * len(positions)
    
    ghost_layer_thickness = num_layers * lattice_param
    box_sizes = [box_bounds[i][1] - box_bounds[i][0] for i in range(3)]
    
    # Identificar átomos en cada región (cerca de caras)
    atoms_by_region = {}
    for i, pos in enumerate(positions):
        region = []
        for dim in range(3):
            dist_from_min = pos[dim] - box_bounds[dim][0]
            dist_from_max = box_bounds[dim][1] - pos[dim]
            
            if dist_from_min < ghost_layer_thickness:
                region.append(f'{dim}_min')
            elif dist_from_max < ghost_layer_thickness:
                region.append(f'{dim}_max')
        
        if region:  # Átomo está cerca de algún borde
            region_key = tuple(sorted(region))
            if region_key not in atoms_by_region:
                atoms_by_region[region_key] = []
            atoms_by_region[region_key].append((i, pos))
    
    # Generar traslaciones para PBC completo
    # Necesitamos replicar en todas las combinaciones de ±1 en cada dimensión
    translations = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue  # No trasladar átomos originales
                translations.append((dx, dy, dz))
    
    ghost_count_by_type = {'caras': 0, 'aristas': 0, 'esquinas': 0}
    
    # Para cada región y cada traslación, crear ghosts si es necesario
    for region_key, atoms in atoms_by_region.items():
        num_dims = len(region_key)  # Cuántas dimensiones están en el borde
        
        if num_dims == 1:
            region_type = 'caras'
        elif num_dims == 2:
            region_type = 'aristas'
        else:  # num_dims == 3
            region_type = 'esquinas'
        
        # Para cada átomo en esta región
        for orig_idx, orig_pos in atoms:
            # Determinar qué traslaciones son necesarias
            needed_shifts = []
            
            for dim_side in region_key:
                dim = int(dim_side[0])
                side = dim_side[2:]  # 'min' o 'max'
                
                if side == 'min':
                    # Átomo cerca del mínimo → replicar hacia el máximo
                    needed_shifts.append((dim, +1))
                else:
                    # Átomo cerca del máximo → replicar hacia el mínimo
                    needed_shifts.append((dim, -1))
            
            # Generar todas las combinaciones de shifts
            if num_dims == 1:
                # CARA: solo un shift
                dim, direction = needed_shifts[0]
                ghost_pos = orig_pos.copy()
                ghost_pos[dim] += direction * box_sizes[dim]
                all_positions.append(ghost_pos)
                ghost_map.append(orig_idx)
                is_ghost.append(True)
                ghost_count_by_type['caras'] += 1
            
            elif num_dims == 2:
                # ARISTA: shifts individuales + combinado
                dim1, dir1 = needed_shifts[0]
                dim2, dir2 = needed_shifts[1]
                
                # Shift 1
                ghost_pos1 = orig_pos.copy()
                ghost_pos1[dim1] += dir1 * box_sizes[dim1]
                all_positions.append(ghost_pos1)
                ghost_map.append(orig_idx)
                is_ghost.append(True)
                
                # Shift 2
                ghost_pos2 = orig_pos.copy()
                ghost_pos2[dim2] += dir2 * box_sizes[dim2]
                all_positions.append(ghost_pos2)
                ghost_map.append(orig_idx)
                is_ghost.append(True)
                
                # Shift combinado (diagonal)
                ghost_pos3 = orig_pos.copy()
                ghost_pos3[dim1] += dir1 * box_sizes[dim1]
                ghost_pos3[dim2] += dir2 * box_sizes[dim2]
                all_positions.append(ghost_pos3)
                ghost_map.append(orig_idx)
                is_ghost.append(True)
                
                ghost_count_by_type['aristas'] += 3
            
            else:  # num_dims == 3
                # ESQUINA: todos los shifts individuales + combinaciones
                dim1, dir1 = needed_shifts[0]
                dim2, dir2 = needed_shifts[1]
                dim3, dir3 = needed_shifts[2]
                
                # 3 shifts individuales
                for dim, direction in needed_shifts:
                    ghost_pos = orig_pos.copy()
                    ghost_pos[dim] += direction * box_sizes[dim]
                    all_positions.append(ghost_pos)
                    ghost_map.append(orig_idx)
                    is_ghost.append(True)
                
                # 3 shifts dobles (aristas)
                ghost_pos12 = orig_pos.copy()
                ghost_pos12[dim1] += dir1 * box_sizes[dim1]
                ghost_pos12[dim2] += dir2 * box_sizes[dim2]
                all_positions.append(ghost_pos12)
                ghost_map.append(orig_idx)
                is_ghost.append(True)
                
                ghost_pos13 = orig_pos.copy()
                ghost_pos13[dim1] += dir1 * box_sizes[dim1]
                ghost_pos13[dim3] += dir3 * box_sizes[dim3]
                all_positions.append(ghost_pos13)
                ghost_map.append(orig_idx)
                is_ghost.append(True)
                
                ghost_pos23 = orig_pos.copy()
                ghost_pos23[dim2] += dir2 * box_sizes[dim2]
                ghost_pos23[dim3] += dir3 * box_sizes[dim3]
                all_positions.append(ghost_pos23)
                ghost_map.append(orig_idx)
                is_ghost.append(True)
                
                # 1 shift triple (esquina opuesta)
                ghost_pos123 = orig_pos.copy()
                ghost_pos123[dim1] += dir1 * box_sizes[dim1]
                ghost_pos123[dim2] += dir2 * box_sizes[dim2]
                ghost_pos123[dim3] += dir3 * box_sizes[dim3]
                all_positions.append(ghost_pos123)
                ghost_map.append(orig_idx)
                is_ghost.append(True)
                
                ghost_count_by_type['esquinas'] += 7
    
    ghost_positions = np.array(all_positions)
    ghost_map = np.array(ghost_map)
    is_ghost = np.array(is_ghost)
    
    n_ghosts = np.sum(is_ghost)
    
    print(f"\n{'='*80}")
    print(f"RESUMEN GHOST PARTICLES")
    print(f"{'='*80}")
    print(f"  Átomos originales: {len(positions)}")
    print(f"  Ghosts (caras):    {ghost_count_by_type['caras']}")
    print(f"  Ghosts (aristas):  {ghost_count_by_type['aristas']}")
    print(f"  Ghosts (esquinas): {ghost_count_by_type['esquinas']}")
    print(f"  Total ghosts:      {n_ghosts}")
    print(f"  Total para Delaunay: {len(ghost_positions)}")
    print(f"{'='*80}\n")
    
    return ghost_positions, ghost_map, is_ghost


# ============================================================================
# ALPHA SHAPE CON GHOST PARTICLES
# ============================================================================

class AlphaShapeWithGhosts:
    """
    Alpha Shape usando Ghost Particles (técnica de OVITO)
    """
    
    def __init__(self, positions, probe_radius, box_bounds=None,
                 lattice_param=None, num_ghost_layers=2,
                 smoothing_level=0):
        
        self.positions_original = np.array(positions, dtype=np.float64)
        self.probe_radius = probe_radius
        self.smoothing_level = smoothing_level
        self.num_ghost_layers = num_ghost_layers
        
        # Auto-detectar box bounds
        if box_bounds is None:
            box_bounds = tuple(
                (self.positions_original[:, i].min(), 
                 self.positions_original[:, i].max())
                for i in range(3)
            )
        self.box_bounds = box_bounds
        
        # Auto-detectar parámetro de red
        if lattice_param is None:
            self.lattice_param = detect_lattice_parameter(self.positions_original)
        else:
            self.lattice_param = lattice_param
        
        # Resultados
        self.ghost_positions = None
        self.ghost_map = None
        self.is_ghost = None
        self.surface_vertices = None
        self.surface_faces = None
        self.surface_area = None
        self._surface_atom_indices = None
    
    def perform(self):
        """Ejecutar Alpha Shape con ghost particles"""
        
        print(f"\n{'='*80}")
        print(f"ALPHA SHAPE CON GHOST PARTICLES")
        print(f"{'='*80}")
        print(f"\nParámetros:")
        print(f"  - probe_radius: {self.probe_radius}")
        print(f"  - lattice_param: {self.lattice_param:.4f}")
        print(f"  - num_ghost_layers: {self.num_ghost_layers}")
        print(f"  - smoothing_level: {self.smoothing_level}")
        
        # PASO 1: Crear ghost particles
        self.ghost_positions, self.ghost_map, self.is_ghost = create_ghost_layers(
            self.positions_original,
            self.box_bounds,
            self.lattice_param,
            self.num_ghost_layers
        )
        
        # PASO 2: Delaunay con ghost particles
        print("\n[1/4] Generando Delaunay tessellation (con ghosts)...")
        delaunay = Delaunay(self.ghost_positions)
        print(f"  ✓ Tetraedros: {len(delaunay.simplices)}")
        
        # PASO 3: Filtrar tetraedros por alpha
        print(f"\n[2/4] Filtrando tetraedros (alpha={self.probe_radius**2:.2f})...")
        valid_tets = self._filter_tetrahedra(delaunay)
        print(f"  ✓ Válidos: {len(valid_tets)}/{len(delaunay.simplices)} "
              f"({100*len(valid_tets)/len(delaunay.simplices):.1f}%)")
        
        # PASO 4: Extraer superficie
        print("\n[3/4] Extrayendo facetas de superficie...")
        surface_facets = self._extract_surface_facets(delaunay, valid_tets)
        print(f"  ✓ Facetas: {len(surface_facets)}")
        
        # PASO 5: Construir malla (SOLO con átomos reales)
        print("\n[4/4] Construyendo malla (filtrando ghosts)...")
        self.surface_vertices, self.surface_faces = self._build_mesh(
            delaunay, surface_facets
        )
        print(f"  ✓ Vértices (reales): {len(self.surface_vertices)}")
        print(f"  ✓ Caras: {len(self.surface_faces)}")
        
        # Suavizado
        if self.smoothing_level > 0:
            print(f"\n[Suavizado] {self.smoothing_level} iteraciones...")
            self.surface_vertices = self._smooth_mesh(
                self.surface_vertices, self.surface_faces, self.smoothing_level
            )
        
        # Área
        self.surface_area = self._compute_surface_area()
        
        print(f"\n{'='*80}")
        print(f"✓ COMPLETADO")
        print(f"  Área de superficie: {self.surface_area:.4f}")
        print(f"  Átomos superficiales: {len(self._surface_atom_indices)}")
        print(f"{'='*80}\n")
        
        return self
    
    def _filter_tetrahedra(self, delaunay):
        """Filtrar tetraedros por circumradius"""
        valid_tets = []
        for tet_idx, tet in enumerate(delaunay.simplices):
            verts = self.ghost_positions[tet]
            circumradius = self._compute_circumradius(verts)
            if circumradius <= self.probe_radius:
                valid_tets.append(tet_idx)
        return np.array(valid_tets)
    
    def _compute_circumradius(self, vertices):
        """Calcular circumradius de tetraedro"""
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
        """Extraer facetas de superficie"""
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
        """
        Construir malla SOLO con átomos REALES (sin ghosts)
        """
        if not surface_facets:
            self._surface_atom_indices = np.array([], dtype=int)
            return np.array([]), np.array([])
        
        # Extraer índices únicos de vértices superficiales
        all_surface_indices = set()
        for facet in surface_facets:
            all_surface_indices.update(facet)
        
        # FILTRAR: mantener solo índices de átomos REALES (no ghosts)
        # Los átomos reales tienen índices < len(positions_original)
        n_real_atoms = len(self.positions_original)
        real_surface_indices = [idx for idx in all_surface_indices 
                                if idx < n_real_atoms]
        
        real_surface_indices = sorted(real_surface_indices)
        
        print(f"    - Vértices totales en superficie: {len(all_surface_indices)}")
        print(f"    - Vértices ghost: {len(all_surface_indices) - len(real_surface_indices)}")
        print(f"    - Vértices reales: {len(real_surface_indices)}")
        
        if not real_surface_indices:
            self._surface_atom_indices = np.array([], dtype=int)
            return np.array([]), np.array([])
        
        # Guardar índices de átomos superficiales (en array original)
        self._surface_atom_indices = np.array(real_surface_indices, dtype=int)
        
        # Crear mapeo de índices
        vertex_map = {old_idx: new_idx 
                     for new_idx, old_idx in enumerate(real_surface_indices)}
        
        # Extraer vértices (de posiciones originales)
        vertices = self.positions_original[real_surface_indices]
        
        # Remapear caras (solo caras que tienen todos sus vértices reales)
        faces = []
        for facet in surface_facets:
            # Verificar si todos los vértices son reales
            if all(v < n_real_atoms for v in facet):
                remapped_facet = [vertex_map[v] for v in facet]
                faces.append(remapped_facet)
        
        print(f"    - Caras válidas (solo vértices reales): {len(faces)}")
        
        return vertices, np.array(faces)
    
    def _smooth_mesh(self, vertices, faces, iterations):
        """Suavizado Laplaciano"""
        smoothed_vertices = vertices.copy()
        
        for iteration in range(iterations):
            new_vertices = smoothed_vertices.copy()
            vertex_neighbors = [set() for _ in range(len(smoothed_vertices))]
            
            for face in faces:
                for i in range(len(face)):
                    for j in range(len(face)):
                        if i != j:
                            vertex_neighbors[face[i]].add(face[j])
            
            for v_idx in range(len(smoothed_vertices)):
                neighbors = list(vertex_neighbors[v_idx])
                if neighbors and len(neighbors) > 2:
                    new_vertices[v_idx] = smoothed_vertices[neighbors].mean(axis=0)
            
            smoothed_vertices = new_vertices
        
        return smoothed_vertices
    
    def _compute_surface_area(self):
        """Calcular área de superficie"""
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
    
    def get_surface_atoms_indices(self):
        """Obtener índices de átomos superficiales (en array original)"""
        return self._surface_atom_indices if self._surface_atom_indices is not None else np.array([], dtype=int)


# ============================================================================
# PARSER LAMMPS
# ============================================================================

class LAMMPSDumpParser:
    """Parser para archivos LAMMPS dump"""
    
    @staticmethod
    def read(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        data = {
            'timestep': 0,
            'n_atoms': 0,
            'box_bounds': None,
            'columns': [],
            'atoms': OrderedDict(),
            'positions': None,
            'atom_ids_ordered': []
        }
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line == "ITEM: TIMESTEP":
                data['timestep'] = int(lines[i+1].strip())
                i += 2
            
            elif line == "ITEM: NUMBER OF ATOMS":
                data['n_atoms'] = int(lines[i+1].strip())
                i += 2
            
            elif line.startswith("ITEM: BOX BOUNDS"):
                parts = line.split()
                pbc = parts[3:] if len(parts) > 3 else ['pp', 'pp', 'pp']
                bounds = []
                i += 1
                for j in range(3):
                    if i < len(lines):
                        bound_line = lines[i].strip().split()
                        if bound_line and not bound_line[0].startswith("ITEM:"):
                            bounds.append((float(bound_line[0]), float(bound_line[1])))
                            i += 1
                        else:
                            break
                data['box_bounds'] = tuple(bounds)
                data['pbc'] = pbc
                continue
            
            elif line.startswith("ITEM: ATOMS"):
                parts = line.split()
                data['columns'] = parts[2:]
                positions_list = []
                i += 1
                
                while i < len(lines) and not lines[i].startswith("ITEM:"):
                    atom_line = lines[i].strip()
                    if atom_line:
                        values = atom_line.split()
                        atom_dict = {}
                        for col_idx, col_name in enumerate(data['columns']):
                            try:
                                val = float(values[col_idx])
                                if col_name == 'id' or col_name == 'type':
                                    val = int(val)
                            except (ValueError, IndexError):
                                val = values[col_idx]
                            atom_dict[col_name] = val
                        
                        atom_id = int(values[0])
                        data['atoms'][atom_id] = atom_dict
                        data['atom_ids_ordered'].append(atom_id)
                        
                        if 'x' in data['columns']:
                            x_idx = data['columns'].index('x')
                            y_idx = data['columns'].index('y')
                            z_idx = data['columns'].index('z')
                            positions_list.append([
                                float(values[x_idx]),
                                float(values[y_idx]),
                                float(values[z_idx])
                            ])
                    i += 1
                
                data['positions'] = np.array(positions_list)
                continue
            
            i += 1
        
        return data
    
    @staticmethod
    def write(filename, data, filtered_atom_ids):
        with open(filename, 'w') as f:
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{data['timestep']}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{len(filtered_atom_ids)}\n")
            f.write("ITEM: BOX BOUNDS")
            if 'pbc' in data:
                f.write(f" {' '.join(data['pbc'])}\n")
            else:
                f.write(" pp pp pp\n")
            for xmin, xmax in data['box_bounds']:
                f.write(f"{xmin:.6f} {xmax:.6f}\n")
            f.write(f"ITEM: ATOMS {' '.join(data['columns'])}\n")
            for atom_id in filtered_atom_ids:
                if atom_id in data['atoms']:
                    atom = data['atoms'][atom_id]
                    values = []
                    for col in data['columns']:
                        val = atom[col]
                        if isinstance(val, (int, float)):
                            if isinstance(val, float):
                                values.append(f"{val:.8f}")
                            else:
                                values.append(str(int(val)))
                        else:
                            values.append(str(val))
                    f.write(" ".join(values) + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Alpha Shape con Ghost Particles (Técnica OVITO)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input_dump", help="Archivo dump de entrada")
    parser.add_argument("output_dump", help="Archivo dump de salida")
    parser.add_argument("--probe-radius", type=float, default=2.0,
                        help="Radio de la esfera de prueba (default: 2.0)")
    parser.add_argument("--lattice-param", type=float, default=None,
                        help="Parámetro de red (default: auto-detectar)")
    parser.add_argument("--num-ghost-layers", type=int, default=2,
                        help="Número de capas fantasma por cara (default: 2)")
    parser.add_argument("--smoothing", type=int, default=8,
                        help="Nivel de suavizado (default: 8)")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"{'ALPHA SHAPE CON GHOST PARTICLES (OVITO)':^80}")
    print(f"{'='*80}\n")
    
    # Leer
    print(f"[CARGA] Leyendo: {args.input_dump}")
    data = LAMMPSDumpParser.read(args.input_dump)
    print(f"  ✓ Átomos: {len(data['atoms'])}")
    print(f"  ✓ Box: {data['box_bounds']}")
    
    # Procesar
    print(f"\n[PROCESAMIENTO]")
    constructor = AlphaShapeWithGhosts(
        positions=data['positions'],
        probe_radius=args.probe_radius,
        box_bounds=data['box_bounds'],
        lattice_param=args.lattice_param,
        num_ghost_layers=args.num_ghost_layers,
        smoothing_level=args.smoothing
    )
    
    constructor.perform()
    surface_atoms = constructor.get_surface_atoms_indices()
    
    # Exportar
    print(f"\n[EXPORTACIÓN] Escribiendo: {args.output_dump}")
    filtered_ids = sorted([data['atom_ids_ordered'][idx] for idx in surface_atoms])
    LAMMPSDumpParser.write(args.output_dump, data, filtered_ids)
    
    print(f"\n{'='*80}")
    print(f"✓ COMPLETADO")
    print(f"{'='*80}")
    print(f"  Entrada:  {len(data['atoms'])} átomos")
    print(f"  Salida:   {len(filtered_ids)} átomos")
    print(f"  Removidos: {len(data['atoms']) - len(filtered_ids)}")
    print(f"  Área: {constructor.surface_area:.4f}")
    print(f"\n✓ Guardado: {args.output_dump}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
