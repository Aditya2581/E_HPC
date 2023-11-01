from mpi4py import MPI
import numpy as np
from time import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 16
t1 = time()

# Only the rank 0 process generates the array
if rank == 0:
    array = np.random.rand(N)
    # array = np.array([15, 11, 9, 16, 3, 14, 8, 7, 4, 6, 12, 10, 5, 2, 13, 1])
    print(f"size: {np.size(array)}, initial array: {array}")

    # Ensure that the array can be evenly divided among all processes
    remainder = size - (np.size(array) % size)
    z = np.zeros(remainder)
    array = np.append(array, z)
else:
    array = None

t2 = time()

# Ensure that each process gets an equal chunk of the array
extra = size - (N % size)
N = N + extra
local_size = N // size
local_array = np.zeros(local_size)

# Scatter the array to all processes
comm.Scatter(array, local_array, root=0)

# Each process sorts its own chunk of the array
local_array = np.sort(local_array)

t3 = time()

# Odd-even transposition sort
for i in range(size):
    if size == 1:
        # Check for serial computation
        break
    elif i % 2 == 0:
        # Even-phase communication
        if rank % 2 == 0 and rank < size - 1:
            recv_array = comm.sendrecv(local_array, dest=rank + 1, sendtag=0, source=rank + 1, recvtag=0)
            local_array = np.concatenate([local_array, recv_array])
            local_array = np.sort(local_array)
            local_array = local_array[:local_size]
        elif rank % 2 == 1:
            recv_array = comm.sendrecv(local_array, dest=rank - 1, sendtag=0, source=rank - 1, recvtag=0)
            local_array = np.concatenate([recv_array, local_array])
            local_array = np.sort(local_array)
            local_array = local_array[local_size:]
    else:
        # Odd-phase communication
        if rank % 2 == 1 and rank < size - 1:
            recv_array = comm.sendrecv(local_array, dest=rank + 1, sendtag=0, source=rank + 1, recvtag=0)
            local_array = np.concatenate([local_array, recv_array])
            local_array = np.sort(local_array)
            local_array = local_array[:local_size]
        elif rank % 2 == 0 and rank > 0:
            recv_array = comm.sendrecv(local_array, dest=rank - 1, sendtag=0, source=rank - 1, recvtag=0)
            local_array = np.concatenate([recv_array, local_array])
            local_array = np.sort(local_array)
            local_array = local_array[local_size:]

t4 = time()

# Gather all sorted chunks back to rank 0
sorted_array = comm.gather(local_array, root=0)

t5 = time()

# Concatenate all sorted chunks to form the final sorted array
if rank == 0:
    final_array = np.concatenate(sorted_array)[extra:]
    print(f"size: {np.size(final_array)}, final array: {final_array}")
    print("initialization time: {}".format(t2 - t1))
    print("scattering time: {}".format(t3 - t2))
    print("rank odd even time: {}".format(t4 - t3))
    print("gather time: {}".format(t5 - t4))
