import numpy as np
import vqe

def main():
    hamiltonian = np.array([[0, 0, 0, 0], [0, -1, 1, 0], [0, 1, -1, 0], [0, 0, 0, 0]], dtype=np.complex128)
    print(f"Calculating ground state energy for Hamiltonian:\n{hamiltonian}")
    ground_state_energy = vqe.run(hamiltonian)
    print(f"Estimated ground state energy: {ground_state_energy}")

main()
