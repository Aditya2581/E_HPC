from mpi4py import MPI
import numpy as np
from time import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parent_size = 4
if parent_size % np.sqrt(size) != 0.0:
    exit()
p = int(np.sqrt(size))
parent_shape = (parent_size, parent_size)
subarray_size = int(parent_size // p)
subarray_shape = (subarray_size, subarray_size)
A_subarray = np.zeros(subarray_shape, dtype=np.float64)
B_subarray = np.zeros(subarray_shape, dtype=np.float64)

t1 = time()
if rank == 0:
    # Create a large 2D array
    A = np.random.rand(parent_size, parent_size)
    B = np.random.rand(parent_size, parent_size)

    # Create datatypes for each subarray with different starting positions
    for i in range(p):
        for j in range(p):
            subarray_dtype = MPI.DOUBLE.Create_subarray(parent_shape, subarray_shape, (i * subarray_size, j * subarray_size))
            subarray_dtype.Commit()
            dest = p * i + j
            if dest == 0:
                comm.Sendrecv([A, 1, subarray_dtype], dest=dest, sendtag=0, recvbuf=A_subarray, source=0, recvtag=0)
                comm.Sendrecv([B, 1, subarray_dtype], dest=dest, sendtag=1, recvbuf=B_subarray, source=0, recvtag=1)
            else:
                comm.Send([A, 1, subarray_dtype], dest=dest, tag=0)
                comm.Send([B, 1, subarray_dtype], dest=dest, tag=1)
            subarray_dtype.Free()

else:
    comm.Recv([A_subarray, MPI.DOUBLE], source=0, tag=0)
    comm.Recv([B_subarray, MPI.DOUBLE], source=0, tag=1)


# First Left Ring shift for A sub matrices which depends on the "row number"
def first_ring_shift():
    i = int(rank / p)
    if i != 0:
        temp_A = A_subarray.copy()
        source = rank + i
        dest = rank - i
        if source >= (i + 1) * p:
            source = source - p
        if dest < (i * p):
            dest = dest + p
        comm.Sendrecv(temp_A, dest=dest, sendtag=0, recvbuf=A_subarray, source=source, recvtag=0)


# First Up Column shift for B sub matrices which depends on the "column number"
def first_column_shift():
    i = int(rank % p)
    if i != 0:
        temp_B = B_subarray.copy()
        source = rank + p * i
        dest = rank - p * i
        if source > i + (p - 1) * p:
            source = source - p * p
        if dest < i:
            dest = dest + p * p
        comm.Sendrecv(temp_B, dest=dest, sendtag=0, recvbuf=B_subarray, source=source, recvtag=0)


# Left Ring Shift for A sub matrix
def ring_shift():
    i = int(rank / p)
    temp_A = A_subarray.copy()
    source = rank + 1
    dest = rank - 1
    if source >= (i + 1) * p:
        source = source - p
    if dest < (i * p):
        dest = dest + p
    comm.Sendrecv(temp_A, dest=dest, sendtag=0, recvbuf=A_subarray, source=source, recvtag=0)


# Up Column Shift for B sub matrix
def column_shift():
    i = int(rank % p)
    temp_B = B_subarray.copy()
    source = rank + p
    dest = rank - p
    if source > i + (p - 1) * p:
        source = source - p * p
    if dest < i:
        dest = dest + p * p
    comm.Sendrecv(temp_B, dest=dest, sendtag=0, recvbuf=B_subarray, source=source, recvtag=0)


t2 = time()

# Calling functions

first_ring_shift()
first_column_shift()
C_submatrix = np.matmul(A_subarray, B_subarray)
t3 = time()
for _ in range(p - 1):
    ring_shift()
    column_shift()
    C_submatrix += np.matmul(A_subarray, B_subarray)

t4 = time()
C_gathered = comm.gather(C_submatrix, root=0)
t5 = time()

if rank == 0:
    # Since the problem size is small, calculating the product using Parent Matrices (just to compare)
    original = np.matmul(A, B)
    print("A*B = ", original)

    # Arranging the sub-matrices to make them into correct shape
    C_final = None
    for i in range(p):
        temp = C_gathered[i * p]
        for j in range(1, p):
            temp = np.concatenate((temp, C_gathered[i * p + j]), axis=1)
        if C_final is None:
            C_final = temp
        else:
            C_final = np.concatenate((C_final, temp), axis=0)
    print("C_final: ", C_final)
    if (abs(original - C_final) < 10**-8).all():
        print("Same Values")
    else:
        print("Different Values")
    print("Time to create parent and divide into subarrays: ", t2 - t1)
    print("Time taken to do the first iteration", t3 - t2)
    print("Time taken to do the rest of the iterations: ", t4 - t3)
    print("Time taken to gather: ", t5 - t4)
