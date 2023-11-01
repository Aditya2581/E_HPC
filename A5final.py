# Import required modules
from mpi4py import MPI
from time import time
import numpy as np

# Initialize MPI communication
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Define parameters for heat equation
rho = 8933
k = 401
c = 383.67
alpha = k / (rho * c)
L = 1

Ui = UL = 300   # initial Temperature and Left Boundary condition
UR = 600        # Right Boundary condition

# Define time and space steps and calculate stability condition
delta_t = 0.01   # time step
delta_x = 0.01   # distance between nodes in the rod
stability = (delta_t * alpha) / (delta_x ** 2)  # condition for stability of the method

N = int(L / delta_x) - 2    # Total number of internal nodes in the rod

# Making sure resources are not wasted
if size > N:
    print("Too many processes, reduce the number of processor")
    exit()

# Checking the stability of the process
if stability < 0.5:
    # Calculate number of points for each process
    if rank != size-1:
        n = N//size
    else:
        n = N-(size-1)*(N//size)

    # Initialize temperature array for each process
    U = np.ones(n) * Ui

    # Define convergence threshold
    delta_U = 10 ** -6

    # Initialize variables for loop
    U_last = U.copy()
    start_time = time()
    itr = 0
    max_error = 100

    # Loop until convergence
    while max_error > delta_U:
        T_left = 300.0
        T_right = 600.0

        # Send temperature values for left boundary condition to the left process
        if rank != 0:
            comm.send(U[0], dest=rank-1)
            T_left = comm.recv(source=rank - 1)

        # Send temperature values for right boundary condition to the right process
        if rank != size-1:
            comm.send(U[n-1], dest=rank+1)
            T_right = comm.recv(source=rank + 1)

        # Calculate new temperature values based on heat equation
        U = U_last + delta_t * alpha * (np.pad(U_last, (1, 1), constant_values=(T_left, T_right))[2:] - 2 * U_last + np.pad(U_last, (1, 1), constant_values=(T_left, T_right))[:n]) / (delta_x ** 2)

        # Calculate maximum error and check for convergence
        error = abs(max(U - U_last))
        max_error = comm.allreduce(error, op=MPI.MAX)
        U_last = U.copy()
        itr += 1

    end_time = time()

    # Gather final temperature values from all processes
    # U_gather = comm.gather(U, root=0)

    # Print final temperature values and execution time
    if rank == 0:
        # U_final = np.concatenate(U_gather)[:]
        # print(np.pad(U_final, (1, 1), constant_values=(300, 600)))
        print(end_time - start_time)

# If stability condition is not met, print error message
else:
    print("Change parameters to ensure stability")
