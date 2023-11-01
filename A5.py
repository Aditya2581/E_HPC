from mpi4py import MPI
from time import time
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
rho = 8933
k = 401
c = 383.67
alpha = k / (rho * c)
L = 1
Ui = UL = 300
UR = 600

delta_t = 0.1
delta_x = 0.1

stability = (delta_t * alpha) / (delta_x ** 2)
N = int(L / delta_x) - 2

if stability < 0.5:
    if rank != size-1:
        n = N//size
    else:
        n = N-(size-1)*(N//size)
    U = np.ones(n) * Ui

    delta_U = 10 ** -6

    U_last = U.copy()

    start_time = time()
    itr = 0
    max_error = 100
    error = 100
    while max_error > delta_U:
        T_left = 300.0
        T_right = 600.0
        # for Left BC
        if rank != 0:
            comm.send(U[0], dest=rank-1)
            T_left = comm.recv(source=rank - 1)
        # for Right BC
        if rank != size-1:
            comm.send(U[n-1], dest=rank+1)
            T_right = comm.recv(source=rank + 1)

        U = U_last + delta_t * alpha * (np.pad(U_last, (1, 1), constant_values=(T_left, T_right))[2:] - 2 * U_last + np.pad(U_last, (1, 1), constant_values=(T_left, T_right))[:n]) / (delta_x ** 2)

        error = abs(max(U - U_last))
        max_error = comm.allreduce(error, op=MPI.MAX)
        U_last = U.copy()
        itr += 1

    end_time = time()
    U_gather = comm.gather(U, root=0)
    if rank == 0:
        U_final = np.concatenate(U_gather)[:]
        print(np.pad(U_final, (1, 1), constant_values=(300, 600)))
        print(end_time - start_time, itr)
else:
    print("Change parameters to ensure stability")
