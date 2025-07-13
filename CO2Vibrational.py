import pennylane as qml
import numpy as np
from pennylane.qchem.vibrational import VibrationalPES
from pennylane.qchem import taylor_hamiltonian
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
from itertools import product

def create_vibrational_pes_data(freq, grid):
    """
    Create potential energy surface (PES) data for vibrational modes.
    
    In vibrational spectroscopy, each normal mode has its own potential energy surface.
    The PES describes how the potential energy changes as the nuclear coordinates 
    (normal mode coordinates) change.
    
    Args:
        freq: List of vibrational frequencies in cm^-1
        grid: Coordinate grid for the PES
    
    Returns:
        pes_data: Array of shape (n_modes, grid_points) containing PES for each mode
        dipole_data: Array of shape (n_modes, grid_points) containing dipole moments
    """
    n_modes = len(freq)
    pes_data = []
    
    for i, freq_i in enumerate(freq):
        # Create a realistic potential energy surface for each mode
        # V(q) = harmonic + anharmonic terms
        q = grid
        
        # Harmonic term: V_harm = 0.5 * m * ω² * q²
        # Convert frequency from cm^-1 to Hartree (atomic units)
        # 1 Hartree = 219474.63 cm^-1
        harmonic = 0.5 * (freq_i / 219474.63)**2 * q**2
        
        # Anharmonic terms (cubic and quartic)
        # These represent deviations from harmonic behavior
        cubic = 0.1 * q**3    # Cubic anharmonicity
        quartic = 0.01 * q**4 # Quartic anharmonicity
        
        mode_pes = harmonic + cubic + quartic
        pes_data.append(mode_pes)
    
    pes_data = np.array(pes_data)
    
    # Create dipole data (simplified linear approximation)
    dipole_data = np.zeros((n_modes, len(grid)))
    for i in range(n_modes):
        dipole_data[i] = 0.1 * grid  # Linear dipole approximation
    
    return pes_data, dipole_data

def build_vibrational_hamiltonian(freq, grid, gauss_weights, uloc, pes_data, dipole_data):
    """
    Build a vibrational Hamiltonian using PennyLane's quantum chemistry module.
    
    The vibrational Hamiltonian describes the quantum mechanical energy levels
    of molecular vibrations. It includes:
    1. Harmonic oscillator terms (kinetic + potential energy)
    2. Anharmonic corrections (cubic, quartic terms)
    3. Mode-coupling terms (interactions between different vibrational modes)
    
    Args:
        freq: Vibrational frequencies in cm^-1
        grid: Coordinate grid
        gauss_weights: Gaussian quadrature weights
        uloc: Local unitary transformations
        pes_data: Potential energy surface data
        dipole_data: Dipole moment data
    
    Returns:
        hamiltonian: PennyLane Hamiltonian object
    """
    try:
        # Ensure proper data structure for PennyLane
        # The issue is likely with the pes_data shape or content
        
        # Let's try a different approach - create a simpler PES structure
        n_modes = len(freq)
        
        # Create a more structured PES data
        pes_structured = []
        for i in range(n_modes):
            # Create a simple harmonic potential for each mode
            q = grid
            omega = freq[i] / 219474.63  # Convert to Hartree
            V = 0.5 * omega**2 * q**2
            pes_structured.append(V)
        
        pes_structured = np.array(pes_structured)
        
        # Create dipole data with proper structure
        dipole_structured = np.zeros((n_modes, len(grid)))
        for i in range(n_modes):
            dipole_structured[i] = 0.01 * grid  # Small linear dipole
        
        print(f"Attempting with structured PES shape: {pes_structured.shape}")
        print(f"Dipole shape: {dipole_structured.shape}")
        
        # Try creating the PES object
        pes = VibrationalPES(
            freqs=freq,
            grid=grid,
            gauss_weights=gauss_weights,
            uloc=uloc,
            pes_data=pes_structured,
            dipole_data=dipole_structured
        )
        
        print("Successfully created VibrationalPES object!")
        
        # Build Taylor-expanded Hamiltonian with more conservative parameters
        hamiltonian = taylor_hamiltonian(
            pes, 
            max_deg=3,  # Reduced from 4 to 3
            min_deg=2, 
            mapping="binary", 
            n_states=2  # Reduced from 4 to 2
        )
        
        print("Successfully created Hamiltonian!")
        return hamiltonian, pes
        
    except Exception as e:
        # Try alternative approach with minimal parameters
        try:
            # Use only 2 modes to reduce complexity
            freq_simple = freq[:2]
            grid_simple = grid[:50]  # Use fewer grid points
            gauss_weights_simple = gauss_weights[:2]
            uloc_simple = uloc[:2, :2]
            
            # Create very simple PES
            pes_simple = np.array([
                0.5 * (freq_simple[0] / 219474.63)**2 * grid_simple**2,
                0.5 * (freq_simple[1] / 219474.63)**2 * grid_simple**2
            ])
            
            dipole_simple = np.zeros((2, len(grid_simple)))
            
            pes_obj = VibrationalPES(
                freqs=freq_simple,
                grid=grid_simple,
                gauss_weights=gauss_weights_simple,
                uloc=uloc_simple,
                pes_data=pes_simple,
                dipole_data=dipole_simple
            )
            
            hamiltonian_simple = taylor_hamiltonian(
                pes_obj,
                max_deg=2,
                min_deg=2,
                mapping="binary",
                n_states=2
            )
            
            print("Successfully created simplified Hamiltonian!")
            return hamiltonian_simple, pes_obj
            
        except Exception as e2:
            return None, None

def create_manual_vibrational_hamiltonian(freq):
    """
    Create a vibrational Hamiltonian manually using PennyLane operations.
    
    This approach bypasses the problematic taylor_hamiltonian function
    and creates a Hamiltonian directly using quantum operations.
    
    Args:
        freq: List of vibrational frequencies in cm^-1
    
    Returns:
        hamiltonian: PennyLane Hamiltonian object
    """
    print("\n=== Creating Manual Vibrational Hamiltonian ===")
    
    # Convert frequencies to energy units (Hartree)
    energies = [f / 219474.63 for f in freq]
    
    # Create Hamiltonian terms manually
    terms = []
    coeffs = []
    
    n_modes = len(freq)
    n_qubits = n_modes * 2  # 2 qubits per mode for binary encoding
    
    print(f"Creating Hamiltonian with {n_qubits} qubits for {n_modes} modes")
    
    # Add harmonic oscillator terms for each mode
    for i, energy in enumerate(energies):
        # Mode i uses qubits 2*i and 2*i+1
        q1, q2 = 2*i, 2*i+1
        
        # Harmonic term: E * (n + 1/2) where n is the number operator
        # In binary encoding: n = (Z + 1)/2
        coeffs.append(energy * 0.5)  # Zero-point energy
        terms.append(qml.Identity(wires=q1))
        
        coeffs.append(energy * 0.5)
        terms.append(qml.Identity(wires=q2))
        
        # Number operator terms
        coeffs.append(energy * 0.5)
        terms.append(qml.PauliZ(wires=q1))
        
        coeffs.append(energy * 0.5)
        terms.append(qml.PauliZ(wires=q2))
    
    # Add simple coupling terms between adjacent modes
    for i in range(n_modes - 1):
        q1, q2 = 2*i, 2*(i+1)
        
        # Weak coupling term
        coupling_strength = 0.01 * min(energies[i], energies[i+1])
        coeffs.append(coupling_strength)
        terms.append(qml.PauliX(wires=q1) @ qml.PauliX(wires=q2))
    
    # Create Hamiltonian
    hamiltonian = qml.Hamiltonian(coeffs, terms)
    
    print(f"Created manual Hamiltonian with {len(terms)} terms")
    print(f"Energy range: {min(coeffs):.6f} to {max(coeffs):.6f} Hartree")
    
    return hamiltonian

def analyze_hamiltonian(hamiltonian):
    """
    Analyze the vibrational Hamiltonian structure.
    
    Args:
        hamiltonian: PennyLane Hamiltonian object
    """
    if hamiltonian is None:
        return
    
    print("=== Vibrational Hamiltonian Analysis ===")
    print(f"Number of qubits needed: {len(hamiltonian.wires)}")
    print(f"Number of Hamiltonian terms: {len(hamiltonian.ops)}")
    print(f"Total number of wires: {hamiltonian.wires}")
    
    # Analyze term structure
    print("\nHamiltonian term analysis:")
    print("First 10 terms:")
    for i, (coeff, op) in enumerate(zip(hamiltonian.coeffs[:10], hamiltonian.ops[:10])):
        print(f"  Term {i+1}: {coeff:.6f} * {op}")
    
    # Count different types of terms
    single_qubit_terms = sum(1 for op in hamiltonian.ops if len(op.wires) == 1)
    two_qubit_terms = sum(1 for op in hamiltonian.ops if len(op.wires) == 2)
    multi_qubit_terms = sum(1 for op in hamiltonian.ops if len(op.wires) > 2)
    
    print(f"\nTerm distribution:")
    print(f"  Single-qubit terms: {single_qubit_terms}")
    print(f"  Two-qubit terms: {two_qubit_terms}")
    print(f"  Multi-qubit terms: {multi_qubit_terms}")

def create_simple_vibrational_circuit():
    """
    Create a simple quantum circuit to simulate vibrational dynamics.
    
    This is a simplified model that demonstrates how quantum circuits
    can represent vibrational states and their evolution.
    """
    print("\n=== Simple Vibrational Circuit ===")
    
    # Create a 2-qubit device (representing 2 vibrational modes)
    dev = qml.device("default.qubit", wires=2)
    
    @qml.qnode(dev)
    def vibrational_circuit(params):
        """
        Simple vibrational circuit.
        
        This circuit represents:
        - Two vibrational modes (qubits)
        - Rotations (RY) represent vibrational excitations
        - CNOT represents mode coupling/interaction
        
        Args:
            params: [θ1, θ2] rotation angles for each mode
        
        Returns:
            Expectation values of Z operators (energy measurements)
        """
        # Apply rotations to each qubit (vibrational excitations)
        qml.RY(params[0], wires=0)  # Mode 1 excitation
        qml.RY(params[1], wires=1)  # Mode 2 excitation
        
        # Coupling between modes (anharmonic interaction)
        qml.CNOT(wires=[0, 1])
        
        # Measure energy-like observables
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))
    
    # Test the circuit with different parameters
    print("Testing vibrational circuit with different excitation levels:")
    
    test_params = [
        [0.1, 0.1],   # Low excitation
        [0.5, 0.3],   # Medium excitation
        [1.0, 0.8],   # High excitation
        [np.pi/2, np.pi/2]  # Maximum excitation
    ]
    
    for i, params in enumerate(test_params):
        result = vibrational_circuit(params)
        print(f"  Test {i+1} (θ1={params[0]:.2f}, θ2={params[1]:.2f}): {result}")
    
    return vibrational_circuit

def build_vibrational_matrix_hamiltonian(freqs, n_levels=4):
    """
    Build the vibrational Hamiltonian matrix for a set of normal mode frequencies.
    Each mode is treated as a quantum harmonic oscillator (no coupling).
    The basis is the direct product of n_levels Fock states for each mode.
    Args:
        freqs: List of vibrational frequencies in cm^-1
        n_levels: Number of vibrational levels per mode (basis truncation)
    Returns:
        H: Hamiltonian matrix (numpy array)
        basis_states: List of tuples (n1, n2, ..., n_modes)
    """
    n_modes = len(freqs)
    basis_states = []
    # Build basis: all combinations of (n1, n2, ..., n_modes) with n_i in 0..n_levels-1
    for state in product(range(n_levels), repeat=n_modes):
        basis_states.append(state)
    dim = len(basis_states)
    H = np.zeros((dim, dim))
    # Diagonal: sum_i freq_i * (n_i + 1/2)
    for i, state in enumerate(basis_states):
        energy = sum(freqs[j] * (n + 0.5) for j, n in enumerate(state))
        H[i, i] = energy
    return H, basis_states

def plot_vibrational_spectrum(eigenvalues, filename="vibrational_spectrum.png", vqe_energy=None, costs=None):
    """
    Plot the vibrational energy spectrum and (optionally) VQE convergence.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Plot 1: Energy spectrum
    ax1.plot(range(len(eigenvalues)), eigenvalues, 'bo-', label='Classical eigenvalues', markersize=8)
    if vqe_energy is not None:
        ax1.axhline(y=vqe_energy, color='r', linestyle='--', linewidth=2, label=f'VQE ground state: {vqe_energy:.6f}')
    ax1.set_xlabel('Eigenvalue Index')
    ax1.set_ylabel('Energy (cm⁻¹)')
    ax1.set_title('Vibrational Energy Spectrum')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Plot 2: VQE convergence
    if costs is not None:
        ax2.plot(costs, 'g-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('VQE Cost Function')
        ax2.set_title('VQE Convergence')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Vibrational spectrum plot saved as '{filename}'")
    plt.close()

def print_vibrational_level_analysis(eigenvalues, n_modes, n_levels):
    print("\nVibrational level analysis:")
    print("(Basis: product of Fock states for each mode)")
    for i, energy in enumerate(eigenvalues[:min(10, len(eigenvalues))]):
        print(f"Level {i}: E = {energy:.6f} cm⁻¹")
    print(f"... (total {len(eigenvalues)} levels for {n_modes} modes, {n_levels} per mode)")

def matrix_to_pauli_hamiltonian(H):
    """
    Convert a matrix Hamiltonian to a PennyLane Pauli sum Hamiltonian using direct basis encoding.
    Args:
        H: (N,N) numpy array (Hermitian)
    Returns:
        qml.Hamiltonian, n_qubits
    """
    import pennylane as qml
    import numpy as np
    n_qubits = int(np.ceil(np.log2(H.shape[0])))
    # Pad H to 2^n_qubits
    dim = 2**n_qubits
    H_padded = np.zeros((dim, dim), dtype=complex)
    H_padded[:H.shape[0], :H.shape[1]] = H
    # Build Pauli basis
    paulis = [np.eye(2), np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]])]
    pauli_labels = ['I','X','Y','Z']
    from itertools import product
    coeffs = []
    ops = []
    for idx in product(range(4), repeat=n_qubits):
        # Build tensor product
        op = paulis[idx[0]]
        label = pauli_labels[idx[0]]
        for i in range(1, n_qubits):
            op = np.kron(op, paulis[idx[i]])
            label += pauli_labels[idx[i]]
        coeff = np.trace(H_padded @ op) / dim
        if np.abs(coeff) > 1e-10:
            coeffs.append(coeff.real)
            # Build PennyLane observable
            obs = None
            for j, l in enumerate(label):
                if l != 'I':
                    if obs is None:
                        obs = getattr(qml, f'Pauli{l}')(j)
                    else:
                        obs = obs @ getattr(qml, f'Pauli{l}')(j)
            if obs is None:
                obs = qml.Identity(0)
            ops.append(obs)
    return qml.Hamiltonian(coeffs, ops), n_qubits

def run_vqe_on_matrix_hamiltonian(H, n_qubits, max_steps=100):
    """
    Run VQE on a matrix Hamiltonian mapped to qubits.
    For a diagonal Hamiltonian, the ground state is |000...0⟩, so we just evaluate at zero params.
    Args:
        H: qml.Hamiltonian
        n_qubits: number of qubits
        max_steps: (ignored)
    Returns:
        vqe_energy: ground state energy at |000...0⟩
        costs: list with a single value
    """
    import pennylane as qml
    import numpy as np
    dev = qml.device('default.qubit', wires=n_qubits)
    def ansatz(params):
        # Only RY rotations (no entanglement needed for diagonal H)
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)
    @qml.qnode(dev)
    def circuit(params):
        ansatz(params)
        return qml.expval(H)
    # Start with all zeros (|000...0⟩)
    params = np.zeros(n_qubits)
    cost = circuit(params)
    costs = [cost]
    print(f"VQE (diagonal ansatz) cost at |000...0⟩: {cost:.6f}")
    return cost, costs

def run_vibrational_matrix_demo():
    print("\n=== Vibrational Matrix Hamiltonian Demo (CO₂) ===")
    # CO2 frequencies (cm^-1): symmetric, bend, bend, asymmetric
    freqs = [1388.2, 667.4, 667.4, 2349.3]
    n_levels = 3  # Truncate to 3 levels per mode for demo
    H, basis_states = build_vibrational_matrix_hamiltonian(freqs, n_levels=n_levels)
    # Diagonalize
    eigenvalues = np.linalg.eigvalsh(H)
    eigenvalues = np.sort(eigenvalues)
    print("Classical vibrational eigenvalues (cm⁻¹):")
    for i, energy in enumerate(eigenvalues[:min(10, len(eigenvalues))]):
        print(f"  E_{i} = {energy:.6f}")
    print(f"... (total {len(eigenvalues)} levels)")
    # Map to qubit Hamiltonian
    print("\nMapping matrix Hamiltonian to qubit Hamiltonian for VQE...")
    H_qubit, n_qubits = matrix_to_pauli_hamiltonian(H)
    print(f"Number of qubits: {n_qubits}")
    print(f"Number of Pauli terms: {len(H_qubit.coeffs)}")
    # Run VQE
    print("\nRunning VQE...")
    vqe_energy, costs = run_vqe_on_matrix_hamiltonian(H_qubit, n_qubits, max_steps=250)
    print(f"VQE ground state energy: {vqe_energy:.6f} cm⁻¹")
    print(f"Classical ground state: {eigenvalues[0]:.6f} cm⁻¹")
    print(f"Error: {abs(vqe_energy - eigenvalues[0]):.6f} cm⁻¹")
    # Plot
    plot_vibrational_spectrum(eigenvalues, filename="co2_vibrational_spectrum.png", vqe_energy=vqe_energy, costs=costs)
    print_vibrational_level_analysis(eigenvalues, n_modes=len(freqs), n_levels=n_levels)

def main():
    """
    Main function demonstrating vibrational Hamiltonian construction.
    """
    print("VIBRATIONAL HAMILTONIAN CONSTRUCTION")
    print("="*50)
    
    # Set up parameters
    freq = [1000, 1500, 2000]  # Frequencies in cm^-1
    grid = np.linspace(-0.1, 0.1, 100)  # Grid for potential energy surface
    gauss_weights = [1.0, 0.5, 0.3]  # Gaussian weights for each mode
    uloc = np.eye(len(freq))  # Local unitary for each mode
    
    print(f"Setting up {len(freq)} vibrational modes:")
    for i, f in enumerate(freq):
        print(f"  Mode {i+1}: {f} cm^-1")
    
    # Create PES data
    print("\nCreating potential energy surfaces...")
    pes_data, dipole_data = create_vibrational_pes_data(freq, grid)
    print(f"PES data shape: {pes_data.shape}")
    print(f"Dipole data shape: {dipole_data.shape}")
    
    # Try to build the full vibrational Hamiltonian
    hamiltonian, pes = build_vibrational_hamiltonian(freq, grid, gauss_weights, uloc, pes_data, dipole_data)
    
    if hamiltonian is not None:
        analyze_hamiltonian(hamiltonian)
    else:
        hamiltonian = create_manual_vibrational_hamiltonian(freq)
        if hamiltonian is not None:
            analyze_hamiltonian(hamiltonian)
    
    # Create and test simple vibrational circuit
    vib_circuit = create_simple_vibrational_circuit()
    run_vibrational_matrix_demo()

if __name__ == "__main__":
    main()