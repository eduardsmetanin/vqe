import numpy as np

# This is based on https://michaelgoerz.net/notes/decomposing-two-qubit-hamiltonians-into-pauli-matrices.html

def run(matrix):
    """ Decomposes given Hermitian 4x4 matrix into sum of Pauli matrices.
    :returns: result as a dictionary. For example:
    {
        'II': -0.5, 'IX': 0.0, 'IY': 0.0, 'IZ': 0.0,
        'XI': 0.0, 'XX': 0.5, 'XY': 0.0, 'XZ': 0.0,
        'YI': 0.0, 'YX': 0.0, 'YY': 0.5, 'YZ': 0.0,
        'ZI': 0.0, 'ZX': 0.0, 'ZY': 0.0, 'ZZ': 0.5
    }
    """
    sx = np.array([[0, 1],  [ 1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j],[1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0],  [0, -1]], dtype=np.complex128)
    id = np.array([[1, 0],  [ 0, 1]], dtype=np.complex128)
    s = [id, sx, sy, sz]
    labels = ['I', 'X', 'Y', 'Z']
    decomposed = {}
    for i in range(4):
        for j in range(4):
            label = labels[i] + labels[j]
            ij = 0.25 * hilbert_schmidt_product(np.kron(s[i], s[j]), matrix)
            decomposed[label] = ij
    return decomposed

def hilbert_schmidt_product(m1, m2):
    trace = (np.dot(m1.conjugate().transpose(), m2)).trace()
    if abs(trace.imag) > 0.00001:
        raise AssertionError("Trace should be a real number")
    return trace.real
