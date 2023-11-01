from mpi4py import MPI
import numpy as np
from time import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 10**8
t1 = time()
if rank == 0:
    array = np.random.rand(N)
    # print("size: {}, initial array: {}".format(np.size(array), array))
    print("size: {}, initial array:".format(np.size(array)))
    remainder = size - (np.size(array) % size)
    z = np.zeros(remainder)
    array = np.append(array, z)
else:
    array = None
t2 = time()
extra = size - (N % size)
N = N + extra

local_size = N // size

local_array = np.zeros(local_size)
comm.Scatter(array, local_array, root=0)
local_array = np.sort(local_array)
t3 = time()
for i in range(size):
    if i % 2 == 0:
        if rank % 2 == 0 and rank < size - 1:
            recv_array = comm.sendrecv(local_array, dest=rank + 1,sendtag=0,source=rank + 1,recvtag=0)
            # recv_array = np.zeros(local_size)
            # comm.Recv(recv_array, source=rank + 1)
            local_array = np.concatenate([local_array, recv_array])
            local_array = np.sort(local_array)
            local_array = local_array[:local_size]
        elif rank % 2 == 1:
            recv_array = comm.sendrecv(local_array, dest=rank - 1,sendtag=0,source=rank - 1,recvtag=0)
            # recv_array = np.zeros(local_size)
            # comm.Recv(recv_array, source=rank - 1)
            local_array = np.concatenate([recv_array, local_array])
            local_array = np.sort(local_array)
            local_array = local_array[local_size:]
    else:
        if rank % 2 == 1 and rank < size - 1:
            recv_array = comm.sendrecv(local_array, dest=rank + 1,sendtag=0,source=rank + 1,recvtag=0)
            # recv_array = np.zeros(local_size)
            # comm.Recv(recv_array, source=rank + 1)
            local_array = np.concatenate([local_array, recv_array])
            local_array = np.sort(local_array)
            local_array = local_array[:local_size]
        elif rank % 2 == 0 and rank > 0:
            recv_array = comm.sendrecv(local_array, dest=rank - 1,sendtag=0,source=rank - 1,recvtag=0)
            # recv_array = np.zeros(local_size)
            # comm.Recv(recv_array, source=rank - 1)
            local_array = np.concatenate([recv_array, local_array])
            local_array = np.sort(local_array)
            local_array = local_array[local_size:]
t4 = time()
sorted_array = comm.gather(local_array, root=0)
t5 = time()
if rank == 0:
    final_array = np.concatenate(sorted_array)[extra:]
    # print("size: {}, final array: {}".format(np.size(final_array), final_array))
    print("size: {}, final array:".format(np.size(final_array)))
    print("initialization time: {}".format(t2-t1))
    print("scattering time: {}".format(t3-t2))
    print("rank odd even time: {}".format(t4-t3))
    print("gather time: {}".format(t5-t4))
