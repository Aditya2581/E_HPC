from mpi4py import MPI
from time import time
import matplotlib.pyplot as plt
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

rho = 8933
k = 401
c = 383.67
alpha = k / (rho * c)
L = 1
Ui = 300

delta_t = 0.1
delta_x = 0.1

stability = (delta_t * alpha) / (delta_x ** 2)

N = int(L / delta_x) - 2
if rank == 0:
    u = np.ones(N) * Ui
else:
    u = None
U = np.empty(N//size)
#error = 100
delta_U = 10 ** -6
comm.Scatter(u, U, root=0)
U_last = U.copy()

# m is time and i is space
start_time = time()
itr = 0
e = 100
while e > delta_U:
    if rank == 0:
        T_left = 300
    else:
        T_left = comm.sendrecv(U[0], dest=rank-1, sendtag=0, source=rank-1, recvtag=0)
        # comm.send(U[0], dest=rank-1)
        # T_left = comm.recv(source=rank - 1)
    if rank == size-1:
        T_right = 600
    else:
        # comm.send(U[N-1], dest=rank+1)
        T_right = comm.sendrecv(U[0], dest=rank+1, sendtag=0, source=rank+1, recvtag=0)
    U = U_last + delta_t * alpha * (np.pad(U_last, (1, 1), constant_values=(T_left, T_right))[2:] - 2 * U_last + np.pad(U_last, (1, 1),constant_values=(T_left, T_right))[:N//size]) / (delta_x ** 2)
    # for i in range(1, N - 1):
    #    U[i] = U[m, i] + delta_t * alpha * (U[m, i + 1] - 2 * U[m, i] + U[m, i - 1]) / (delta_x ** 2)
    comm.barrier()

    error = abs(max(U - U_last))
    e = max(comm.gather(error))
    U_last = U.copy()
    itr += 1
    print(rank, itr, error,e)

end_time = time()
print(np.pad(U,(1,1),constant_values=(300,600)))
print(itr)
print(end_time - start_time)
