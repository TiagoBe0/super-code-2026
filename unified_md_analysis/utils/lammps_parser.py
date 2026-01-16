"""
Parser unificado para archivos LAMMPS dump
Soporta lectura y escritura de archivos .dump con manejo completo de metadata
"""

import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple, Any


class LAMMPSDumpParser:
    """
    Parser robusto para archivos LAMMPS dump

    Soporta:
    - Lectura completa de dumps (timestep, box bounds, PBC, posiciones)
    - Escritura de dumps con filtrado de átomos
    - Manejo de columnas personalizadas
    - Preservación de metadata original
    """

    @staticmethod
    def read(filename: str) -> Dict[str, Any]:
        """
        Lee un archivo LAMMPS dump completo

        Args:
            filename: Path al archivo .dump

        Returns:
            dict con:
                - timestep: int
                - n_atoms: int
                - box_bounds: tuple de 3 (xmin,xmax), (ymin,ymax), (zmin,zmax)
                - pbc: list de condiciones de frontera ['pp', 'pp', 'pp']
                - columns: list de nombres de columnas
                - atoms: OrderedDict {atom_id: {col1: val1, ...}}
                - positions: np.array Nx3 con coordenadas [x, y, z]
                - atom_ids_ordered: list de IDs en orden de aparición
        """
        with open(filename, 'r') as f:
            lines = f.readlines()

        data = {
            'timestep': 0,
            'n_atoms': 0,
            'box_bounds': None,
            'pbc': ['pp', 'pp', 'pp'],
            'columns': [],
            'atoms': OrderedDict(),
            'positions': None,
            'atom_ids_ordered': []
        }

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # TIMESTEP
            if line == "ITEM: TIMESTEP":
                data['timestep'] = int(lines[i+1].strip())
                i += 2
                continue

            # NUMBER OF ATOMS
            elif line == "ITEM: NUMBER OF ATOMS":
                data['n_atoms'] = int(lines[i+1].strip())
                i += 2
                continue

            # BOX BOUNDS (con PBC)
            elif line.startswith("ITEM: BOX BOUNDS"):
                parts = line.split()
                # PBC puede ser: pp (periodic), ff (fixed), etc.
                data['pbc'] = parts[3:] if len(parts) > 3 else ['pp', 'pp', 'pp']

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
                continue

            # ATOMS DATA
            elif line.startswith("ITEM: ATOMS"):
                parts = line.split()
                data['columns'] = parts[2:]  # Ej: ['id', 'type', 'x', 'y', 'z']

                positions_list = []
                i += 1

                # Leer todos los átomos hasta el siguiente ITEM o EOF
                while i < len(lines) and not lines[i].startswith("ITEM:"):
                    atom_line = lines[i].strip()
                    if atom_line:
                        values = atom_line.split()

                        # Crear diccionario de átomo
                        atom_dict = {}
                        for col_idx, col_name in enumerate(data['columns']):
                            try:
                                val = float(values[col_idx])
                                # Convertir id y type a int
                                if col_name in ['id', 'type']:
                                    val = int(val)
                            except (ValueError, IndexError):
                                val = values[col_idx]  # Mantener como string si falla
                            atom_dict[col_name] = val

                        # Guardar átomo
                        atom_id = int(values[0])
                        data['atoms'][atom_id] = atom_dict
                        data['atom_ids_ordered'].append(atom_id)

                        # Extraer posiciones si existen columnas x, y, z
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

                data['positions'] = np.array(positions_list) if positions_list else np.array([])
                continue

            i += 1

        return data

    @staticmethod
    def read_simple(dump_content: str) -> np.ndarray:
        """
        Parser simple para extraer solo posiciones de un string
        Útil para procesamiento rápido sin metadata completa

        Args:
            dump_content: contenido del dump como string

        Returns:
            np.array Nx3 con posiciones [x, y, z]
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
            raise ValueError("Formato dump inválido: no se encontró NUMBER OF ATOMS o ATOMS")

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
            raise ValueError("No se encontraron posiciones atómicas en el dump")

        return positions

    @staticmethod
    def write(filename: str, data: Dict[str, Any], filtered_atom_ids: List[int]) -> None:
        """
        Escribe un archivo LAMMPS dump con átomos filtrados

        Args:
            filename: Path al archivo de salida
            data: diccionario con datos originales (de read())
            filtered_atom_ids: lista de IDs de átomos a escribir
        """
        with open(filename, 'w') as f:
            # TIMESTEP
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{data['timestep']}\n")

            # NUMBER OF ATOMS
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{len(filtered_atom_ids)}\n")

            # BOX BOUNDS
            f.write("ITEM: BOX BOUNDS")
            if 'pbc' in data:
                f.write(f" {' '.join(data['pbc'])}\n")
            else:
                f.write(" pp pp pp\n")

            for xmin, xmax in data['box_bounds']:
                f.write(f"{xmin:.6f} {xmax:.6f}\n")

            # ATOMS
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

    @staticmethod
    def write_simple(filename: str, positions: np.ndarray, timestep: int = 0,
                     box_bounds: Tuple[Tuple[float, float], ...] = None) -> None:
        """
        Escribe un dump simple con solo posiciones

        Args:
            filename: Path al archivo de salida
            positions: np.array Nx3 con coordenadas
            timestep: timestep del dump
            box_bounds: ((xmin,xmax), (ymin,ymax), (zmin,zmax)) o None para auto-calcular
        """
        n_atoms = len(positions)

        # Auto-calcular box bounds si no se proveen
        if box_bounds is None:
            mins = positions.min(axis=0)
            maxs = positions.max(axis=0)
            margin = 5.0  # Margen de 5 Angstroms
            box_bounds = tuple((mins[i] - margin, maxs[i] + margin) for i in range(3))

        with open(filename, 'w') as f:
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{timestep}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{n_atoms}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            for xmin, xmax in box_bounds:
                f.write(f"{xmin:.6f} {xmax:.6f}\n")
            f.write("ITEM: ATOMS id type x y z\n")

            for i, pos in enumerate(positions, start=1):
                f.write(f"{i} 1 {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")
