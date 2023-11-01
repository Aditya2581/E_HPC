from mpi4py import MPI
from time import time
import numpy as np
import matplotlib.pyplot as plt

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = time()

# Define the sphere
sphere_center = np.array([0, 0, 10])
sphere_radius = 10

# Define the plane
plane_normal = np.array([0, 1, 0])
plane_distance = 50

# Define the camera
camera_position = np.array([0, 0, 0])
camera_direction = np.array([0, 0, 1])

# Define the image
image_width = 512
image_height = 512
resolution = 1

# Calculate the number of pixels to render on this processor
num_pixels = image_width * image_height
chunk_size = num_pixels // size
start = rank * chunk_size
end = (rank + 1) * chunk_size

# Initialize the image buffer
image_buffer = np.zeros((chunk_size, 3))

# Loop over the pixels assigned to this processor
for i, pixel_index in enumerate(range(start, end)):
    # Calculate the position of the pixel in 3D space
    x = (pixel_index % image_width - image_width / 2) * resolution
    y = (pixel_index // image_width - image_height / 2) * resolution
    pixel_position = np.array([x, y, 1])

    # Cast a ray from the camera through the pixel
    ray_direction = pixel_position - camera_position
    ray_direction /= np.linalg.norm(ray_direction)
    ray = (camera_position, ray_direction)

    # Determine the color of the pixel by tracing the ray through the scene
    color = np.zeros(3)
    intersection_distance = float('inf')
    intersection_point = None
    intersection_normal = None

    # Check for intersection with the sphere
    oc = ray[0] - sphere_center
    a = np.dot(ray[1], ray[1])
    b = 2.0 * np.dot(oc, ray[1])
    c = np.dot(oc, oc) - sphere_radius * sphere_radius
    discriminant = b * b - 4 * a * c

    if discriminant > 0:
        root = np.sqrt(discriminant)
        t = (-b - root) / (2 * a)
        if 0 < t < intersection_distance:
            intersection_distance = t
            intersection_point = ray[0] + t * ray[1]
            intersection_normal = (intersection_point - sphere_center) / sphere_radius

    # Check for intersection with the plane
    t = - (np.dot(ray[0], plane_normal) + plane_distance) / np.dot(ray[1], plane_normal)
    if 0 < t < intersection_distance:
        intersection_distance = t
        intersection_point = ray[0] + t * ray[1]
        intersection_normal = plane_normal

    # Calculate the color of the pixel
    if intersection_point is not None:
        color = np.array([1, 1, 1])
        image_buffer[i] = color

# Gather the image buffer from all processors
image_buffers = comm.gather(image_buffer, root=0)

end_time = time()

# Combine the image buffers into a single image on the root processor
if rank == 0:
    print(end_time-start_time)
    image = np.vstack(image_buffers)
    image = np.reshape(image, (image_height, image_width, 3))
    plt.imshow(image)
    plt.show()
