from mpi4py import MPI
import numpy as np
import random
from time import time

tosses = 10 ** 8  # number of points to be generated

start = time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
N = comm.Get_size()

dtosses = int(tosses / N)  # dividing the points based on the number of processors using

a = 2 * np.random.rand(dtosses, 2)  # generating random points in a square space
b = np.array([1, 1])  # the center of the square and circle

dist = np.sqrt(np.sum((a - b) ** 2, axis=1))  # distance of points from the center of the circle
inside = np.sum(dist < 1)  # points inside the circle
total = np.size(dist)  # total number of points
probab = inside / total  # probability for individual processor
end = time()

timetaken = end - start  # time taken to run the calculation part for each processor

# the if else statement is to send and recieve the time taken and probability
if rank == 0:
    final_probab = probab
    avgtime = timetaken
    for i in range(1, N):
        rprobab = comm.recv(source=i, tag=9)  # recieveing the probability from all other ranks
        avgtime += comm.recv(source=i, tag=4)  # recieveing the time from all other ranks
        final_probab += rprobab
    final_probab = final_probab / N  # Average probability from all the processors
    avgtime = avgtime / N  # Average time taken by each processor
    print("Probability: {}\nAverage Time: {}".format(final_probab, avgtime))
else:
    comm.send(probab, dest=0, tag=9)  # sending the probability to rank 0
    comm.send(timetaken, dest=0, tag=4)  # sending the time to rank 0
