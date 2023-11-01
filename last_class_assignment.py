# Class Assignment submission by Aditya (ME19B072)

import numpy as np
from mpi4py import MPI

# Initialize the MPI environment
comm = MPI.COMM_WORLD

# Get the number of processes
size = comm.Get_size()

# Get the rank of the process
rank = comm.Get_rank()

# Create a 2D Cartesian topology
dims = (4, 4)
periods = (True, True)
reorder = False
cart_comm = comm.Create_cart(dims, periods, reorder)

# Get the source and destination ranks for the left shift operation
lsrc, ldest = cart_comm.Shift(1, -1)
# Get the source and destination ranks for the up shift operation
usrc, udest = cart_comm.Shift(0, -1)

data = 2 * rank
gather_before = comm.gather(data, root=0)

# Perform the left shift operation
left_shift_data = cart_comm.sendrecv(data, dest=ldest, source=lsrc)
gather_after_left = comm.gather(left_shift_data, root=0)

# Perform the up shift operation
up_shift_data = cart_comm.sendrecv(data, dest=udest, source=usrc)
gather_after_up = comm.gather(up_shift_data, root=0)

if rank == 0:
    print(f"before: \n{np.array(gather_before).reshape(dims)}\n")
    print(f"after left shift: \n{np.array(gather_after_left).reshape(dims)}\n")
    print(f"after left then up: \n{np.array(gather_after_up).reshape(dims)}\n")
