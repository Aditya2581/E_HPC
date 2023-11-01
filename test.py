from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    a = 57
    b = comm.sendrecv(a, dest=2, sendtag=0, source=1, recvtag=0)
elif rank == 1:
    a = 35
    b = 89
    comm.send(b, dest=0, tag=0)
elif rank == 2:
    b = 4
    a = comm.recv(source=0, tag=0)

print(rank, a, b)
