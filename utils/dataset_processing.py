import jax
import jax.numpy as jnp
import ase
import numpy as np


def create_jax_structures(ase_atoms_structures, all_species, cutoff_radius):

    # This function essentially takes a list of ase.Atoms objects and converts
    # all the relevant data into a dictionary of jax data structures

    jax_structures = {}
        
    jax_structures["n_structures"] = len(ase_atoms_structures)

    positions = []
    cells = []
    structure_indices = []
    atomic_species = []
    pbcs = []

    for structure_index, atoms in enumerate(ase_atoms_structures):
        positions.append(atoms.positions)
        cells.append(atoms.cell)
        atomic_species_structure = atoms.get_atomic_numbers()
        atomic_species.append(atomic_species_structure)
        for atom_index in range(atoms.positions.shape[0]):
            structure_indices.append(structure_index)
        pbcs.append(atoms.pbc)

    structure_offsets = np.cumsum([0] + [len(structure) for structure in ase_atoms_structures])
    cells = np.array(cells)
    positions = np.concatenate(positions, axis=0)
    structure_indices = np.array(structure_indices)
    atomic_species = np.concatenate(atomic_species, axis=0)
    neighbor_list, batched_neighbor_list_structure_offsets = get_batched_neighbor_list(ase_atoms_structures, cutoff_radius)

    atomic_indices_per_element = {}
    for a_i in all_species:
        atomic_indices_per_element[a_i] = jnp.array(np.nonzero(atomic_species==a_i)[0])

    # Precompute cell shifts for each neighbor pair. 
    # If we just want gradients wrt positions, this can live outside the model.
    cell_shifts = get_cell_shifts(batched_neighbor_list_structure_offsets, jax_structures["n_structures"], neighbor_list, cells)

    # Transform everything into jax arrays (probably on GPU):
    jax_structures["structure_offsets"] = jnp.array(structure_offsets)
    jax_structures["positions"] = jnp.array(positions)
    jax_structures["cells"] = jnp.array(cells)
    jax_structures["structure_indices"] = jnp.array(structure_indices)
    jax_structures["atomic_species"] = jnp.array(atomic_species)
    jax_structures["neighbor_list"] = jnp.array(neighbor_list)
    jax_structures["batched_neighbor_list_structure_offsets"] = jnp.array(batched_neighbor_list_structure_offsets)
    jax_structures["cell_shifts"] = cell_shifts
    jax_structures["atomic_indices_per_element"] = atomic_indices_per_element
    
    # jax_structures["pbcs"] = ...
    return jax_structures


# @jax.jit
def get_cartesian_vectors(positions, jax_structures):

    neighbor_list = jax_structures["neighbor_list"]
    structure_numbers = neighbor_list[:, 0]
    centers = neighbor_list[:, 1]
    neighbors = neighbor_list[:, 2]
    structure_offsets = jax_structures["structure_offsets"]
    cell_shifts = jax_structures["cell_shifts"]

    cartesian_vectors = positions[structure_offsets[structure_numbers]+neighbors] - positions[structure_offsets[structure_numbers]+centers] + cell_shifts
    return cartesian_vectors


def get_cell_shifts(nl_structure_offsets, n_structures, neighbor_list, cells):

    unit_cell_shifts = []
    for i_structure in range(n_structures):
        neighbor_list_i_structure = neighbor_list[nl_structure_offsets[i_structure]:nl_structure_offsets[i_structure+1]]
        cell_i_structure = cells[i_structure]
        unit_cell_shift_vectors = neighbor_list_i_structure[:, -3:]
        unit_cell_shifts.append(
            unit_cell_shift_vectors @ cell_i_structure  # Warning: it works but in a weird way when there is no cell
        )
    unit_cell_shifts = jnp.concatenate(unit_cell_shifts, axis=0)
    return unit_cell_shifts


def get_batched_neighbor_list(ase_structures, cutoff_radius):

    batched_neighbor_list = []
    batched_neighbor_list_structure_offsets = []
    for structure_index in range(len(ase_structures)):
        centers, neighbors, unit_cell_shift_vectors = get_neighbor_list(
            ase_structures[structure_index],
            cutoff_radius
        )
        species = ase_structures[structure_index].get_atomic_numbers()
        structure_neighbor_list = jnp.stack([
            jnp.array([structure_index]*len(centers)), 
            centers, 
            neighbors, 
            species[centers], 
            species[neighbors],
            unit_cell_shift_vectors[:, 0],
            unit_cell_shift_vectors[:, 1],
            unit_cell_shift_vectors[:, 2]
        ], axis=-1)
        batched_neighbor_list.append(structure_neighbor_list)
        batched_neighbor_list_structure_offsets.append(structure_neighbor_list.shape[0])

    batched_neighbor_list = jnp.concatenate(batched_neighbor_list, axis=0)
    batched_neighbor_list_structure_offsets = jnp.cumsum(jnp.array([0] + batched_neighbor_list_structure_offsets))
    return batched_neighbor_list, batched_neighbor_list_structure_offsets


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
