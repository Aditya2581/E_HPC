# Import required libraries
from mpi4py import MPI
import random
from time import time
import matplotlib.pyplot as plt

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define simulation parameters
P = 10 ** 4  # Number of particles
N = 1000  # Number of simulation iterations

# Specify the id to track since tracking all takes a very long
idstotrack = [40]

# Start timer
starttime = time()

# Initialize dictionary to store particle positions
points = {}

# Create particles and distribute them across processes
if rank == 0:
    for i in range(P):
        pos = random.uniform(0, 100)
        if pos >= 100:
            pos -= 100
        point = {i: [pos, 0, [pos]]}
        # Determine which process the particle belongs to based on its position
        des = int(pos // (100 / size))
        if des != 0:
            comm.send(point, dest=des)
        else:
            points.update(point)
    # Notify other processes that all particles have been sent
    for i in range(size - 1):
        comm.send(0, i + 1)
else:
    # Receive particles from process 0 until all particles have been received
    while True:
        r = comm.recv(source=0)
        if r == 0:
            break
        else:
            points.update(r)

# Get number of particles in each process before and after 1000 iterations
print(f"Process {rank}: {len(points)} particles before simulation")

# Record time it takes to distribute particles
isendtime = time()

# Simulate particle movement for N iterations
for j in range(N):
    # Initialize dictionaries to store particles that move left or right
    Lsend = {}
    Rsend = {}
    toremove = []
    # Update particle positions and store particles that move out of the process's domain
    for id in points.keys():
        points[id][0] += random.uniform(-1, 1)
        if id in idstotrack:
            points[id][2].append(points[id][0])
        points[id][1] += 1
        if points[id][0] >= 100:
            points[id][0] -= 100
            Rsend.update({id: points[id]})
            toremove.append(id)
        elif points[id][0] < 0:
            points[id][0] += 100
            Lsend.update({id: points[id]})
            toremove.append(id)
        elif points[id][0] >= (rank + 1) * 100 / size:
            Rsend.update({id: points[id]})
            toremove.append(id)
        elif points[id][0] < rank * 100 / size:
            Lsend.update({id: points[id]})
            toremove.append(id)
    # Remove particles that have moved out of the process's domain
    for i in toremove:
        points.pop(i)
    # Determine which processes to send particles that move left or right
    if rank == size - 1:
        rdes = 0
        ldes = rank - 1
    elif rank == 0:
        rdes = rank + 1
        ldes = size - 1
    else:
        rdes = rank + 1
        ldes = rank - 1
    # Send particles that move left or right to the appropriate processes
    comm.send(Rsend, dest=rdes)
    comm.send(Lsend, dest=ldes)
    # Receive particles that move left or right from neighboring processes
    points.update(comm.recv(source=MPI.ANY_SOURCE))
    points.update(comm.recv(source=MPI.ANY_SOURCE))
    # Wait for all processes to synchronize
    comm.barrier()

# Record time it takes to simulate particle movement
enditertime = time()

# Get number of particles in each process after and after 1000 iterations
print(f"Process {rank}: {len(points)} particles after simulation")

# Colleting all points from all the ranks and sending it to rank 0
all_points = comm.gather(points, root=0)

# Time taken by each processor to run the iterations (which should be same since using barrier)
# print(f"Process: {rank}, Time Taken: {enditertime - isendtime}\n")
if rank == 0:
    # Merge all particles into one dictionary
    merged_points = {}
    for points in all_points:
        merged_points.update(points)

    # Plot the history positions of particles tracked
    for particle_id in idstotrack:
        history = merged_points[particle_id][2]
        plt.plot(history)
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title(f'History of Particle {particle_id}')
        plt.show()
