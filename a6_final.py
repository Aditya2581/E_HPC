import numpy as np
from mpi4py import MPI
from time import time

comm = MPI.COMM_WORLD
# Get the number of processors
size = comm.Get_size()

# Get the rank of the current process
rank = comm.Get_rank()
n = 4

if rank == 0:
    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4)

    # Check that the matrices are square and their dimensions are compatible
    assert a.shape[0] == a.shape[1] == b.shape[0] == b.shape[1]
    assert a.shape[0] % np.sqrt(size) == 0
else:
    a = None
    b = None
# Divide the matrices into submatrices and distribute them among the processors
sub_a = np.empty((int(n // np.sqrt(size)), int(n // np.sqrt(size))), dtype=np.float64)
sub_b = np.empty((int(n // np.sqrt(size)), int(n // np.sqrt(size))), dtype=np.float64)
comm.Scatter(a, sub_a, root=0)
comm.Scatter(b, sub_b, root=0)

# Perform the multiplication using a ring-like communication pattern
for i in range(int(np.sqrt(size))):
    # Compute the product of the local submatrices
    c_local = np.dot(sub_a, sub_b)
    # Shift the submatrices of A and B by one row and one column, respectively
    sub_a = np.roll(sub_a, -1, axis=0)
    sub_b = np.roll(sub_b, -1, axis=1)
    # Communicate the submatrices of A and B among the processors
    sub_a = comm.bcast(sub_a, root=(rank + np.sqrt(size) - 1) % np.sqrt(size))
    sub_b = comm.bcast(sub_b, root=(rank + 1) % np.sqrt(size))
# Gather the submatrices of C to construct the final matrix product
c = None
if rank == 0:
    c = np.empty((n, n), dtype=np.float64)
C_gather = comm.gather(c, root=0)



