from mpi4py import MPI
from time import time
import numpy as np
from PIL import Image

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class Sphere:
    def __init__(self, center, radius, color):
        self.center = center
        self.radius = radius
        self.color = color
