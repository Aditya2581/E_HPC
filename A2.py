from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 10 ** 6  # number of divisions
a = 0
b = 1
dx = (b - a) / n


# integrand function
def f(x):
    return 1 / (1 + x * x)


# domain is divided but rounding the number to integer sometimes gives less numbers
# making is unequal for all the processors
# this if-else solves that problem and gives the last remaining to last rank
if rank != size - 1:
    divint = int(n / size)
    isum = 0
    for i in range(divint):
        isum += f((i + rank * n / size) * dx)
    isum = isum * 2
    comm.send(isum, dest=size - 1)  # sending the summed value to last rank

else:
    # this is for last rank to add all the remaining values
    # this solves the problem of rounding off error
    divint = n - (size - 1) * int(n / size)
    isum = 0
    for i in range(divint):
        isum += f((i + rank * n / size) * dx)
    isum = isum * 2

    # recieveing all the values from different ranks
    # adding all the values and calculating final integral
    final_sum = 0
    for i in range(0, size - 1):
        rsum = comm.recv(source=i)  # recieveing the values from other ranks
        final_sum += rsum
    final_sum += isum
    final_sum = (dx / 2) * (final_sum - f(0) - f(1))  # subtracting the final
    print("Inegral from a to b = {}".format(final_sum))
