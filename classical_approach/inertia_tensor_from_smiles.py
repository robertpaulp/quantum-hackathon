
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# Atomic masses in unified atomic mass units (u, or amu)
# These values are now used *without conversion to kg*.
# The inertia tensor will thus be in units of amu·Å²
atomic_masses = {
    'H': 1.00784,
    'D': 2.01410,       # Deuterium (²H)
    'T': 3.01605,       # Tritium (³H)
    'He': 4.002602,
    'Li': 6.941,
    'Be': 9.0121831,
    'B': 10.81,
    'C': 12.00000,
    'C13': 13.003355,
    'C14': 14.003242,
    'N': 14.003074,
    'N15': 15.000109,
    'O': 15.994915,
    'O17': 16.999131,
    'O18': 17.999159,
    'F': 18.998403,
    'Ne': 20.1797,
    'Na': 22.989769,
    'Mg': 24.305,
    'Al': 26.981538,
    'Si': 28.085,
    'P': 30.973762,
    'S': 32.06,
    'Cl': 35.45,
    'Ar': 39.948,
    'K': 39.0983,
    'Ca': 40.078,
    'Br': 79.904,
    'I': 126.90447
}

def smiles_to_inertia_tensor(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    inertia_tensor = np.zeros((3, 3))

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        isotope = atom.GetIsotope()
        if isotope:
            symbol = f"{symbol}{isotope}"

        if symbol not in atomic_masses:
            raise ValueError(f"Unsupported atom type or isotope: {symbol}")
        mass = atomic_masses[symbol]
        pos = conf.GetAtomPosition(idx)
        x, y, z = pos.x, pos.y, pos.z

        inertia_tensor += mass * np.array([
            [y**2 + z**2, -x*y, -x*z],
            [-x*y, x**2 + z**2, -y*z],
            [-x*z, -y*z, x**2 + y**2]
        ])

    return inertia_tensor
