import jax.numpy as jnp
import ase
import numpy as np


class JaxStructures:

    # This class essentially takes a list of ase.Atoms objects and converts
    # all the relevant data into jax data structures

    def __init__(self, atoms_list):
        
        self.n_structures = len(atoms_list)

        positions = []
        cells = []
        structure_indices = []
        atomic_species = []
        pbcs = []

        for structure_index, atoms in enumerate(atoms_list):
            positions.append(atoms.positions)
            cells.append(atoms.cell)
            for _ in range(atoms.positions.shape[0]):
                structure_indices.append(structure_index)
            atomic_species.append(atoms.get_atomic_numbers())
            pbcs.append(atoms.pbc)

        self.ase_structures = atoms_list
        self.positions = jnp.array(jnp.concatenate(positions, axis=0))
        self.cells = cells
        self.structure_indices = jnp.array(structure_indices)
        self.atomic_species = atomic_species
        self.pbcs = pbcs


def get_cartesian_vectors(jax_structures, cutoff_radius):

    labels = []
    vectors = []

    for structure_index in range(jax_structures.n_structures):

        centers, neighbors, unit_cell_shift_vectors = get_neighbor_list(
            jax_structures.ase_structures[structure_index],
            cutoff_radius
        )
        where_selected_structure = jnp.where(jax_structures.structure_indices == structure_index)[0]
        
        positions = jax_structures.positions[where_selected_structure]
        cell = jax_structures.cells[structure_index]
        species = jax_structures.atomic_species[structure_index]

        structure_vectors = positions[neighbors] - positions[centers] + unit_cell_shift_vectors @ cell  # Warning: it works but in a weird way when there is no cell
        vectors.append(structure_vectors)
        labels.append(
            jnp.stack([
                jnp.array([structure_index]*len(centers)), 
                centers, 
                neighbors, 
                species[centers], 
                species[neighbors],
                unit_cell_shift_vectors[:, 0],
                unit_cell_shift_vectors[:, 1],
                unit_cell_shift_vectors[:, 2]
            ], axis=-1)) # "structure", "center", "neighbor", "species_center", "species_neighbor", "cell_x", "cell_y", "cell_z"

    vectors = jnp.concatenate(vectors, axis=0)
    labels = jnp.concatenate(labels, axis=0)

    return vectors, labels


def get_neighbor_list(structure, cutoff_radius):

    centers, neighbors, unit_cell_shift_vectors = ase.neighborlist.primitive_neighbor_list(
        quantities="ijS",
        pbc=structure.pbc,
        cell=structure.cell,
        positions=structure.positions,
        cutoff=cutoff_radius,
        self_interaction=True,
        use_scaled_positions=False,
    )

    pairs_to_throw = np.logical_and(centers == neighbors, np.all(unit_cell_shift_vectors == 0, axis=1))
    pairs_to_keep = np.logical_not(pairs_to_throw)

    centers = centers[pairs_to_keep]
    neighbors = neighbors[pairs_to_keep]
    unit_cell_shift_vectors = unit_cell_shift_vectors[pairs_to_keep]

    centers = jnp.array(centers)
    neighbors = jnp.array(neighbors)
    unit_cell_shift_vectors = jnp.array(unit_cell_shift_vectors)

    return centers, neighbors, unit_cell_shift_vectors
