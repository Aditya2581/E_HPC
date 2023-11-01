import numpy as np
from numpy.linalg import norm
from PIL import Image
from time import time

start_time = time()

# Define classes for objects in the scene
class Sphere:
    def __init__(self, center, radius, color):
        self.center = center
        self.radius = radius
        self.color = color

    def intersect(self, ray):
        L = self.center - ray.origin
        tca = L.dot(ray.direction)
        d2 = L.dot(L) - tca * tca
        if d2 > self.radius * self.radius:
            return None
        thc = np.sqrt(self.radius * self.radius - d2)
        t0 = tca - thc
        t1 = tca + thc
        if t0 < 0:
            t0 = t1
        if t0 < 0:
            return None
        hit = ray.origin + t0 * ray.direction
        norm = normalize(hit - self.center)
        return (hit, norm)


class Plane:
    def __init__(self, point, normal, color):
        self.point = point
        self.normal = normal
        self.color = color

    def intersect(self, ray):
        denom = self.normal.dot(ray.direction)
        if abs(denom) > 1e-6:
            t = self.normal.dot(self.point - ray.origin) / denom
            if t >= 0:
                hit = ray.origin + t * ray.direction
                return (hit, self.normal)
        return None


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction


class Light:
    def __init__(self, position, color):
        self.position = position
        self.color = color


# Define utility functions
def normalize(x):
    return x / np.sqrt(np.dot(x, x))


def reflect(i, n):
    return i - 2.0 * np.dot(i, n) * n


# Define the cast_ray function
def cast_ray(ray, objects, lights):
    # Find the closest intersection with an object
    hit, hit_norm, hit_obj = None, None, None
    for obj in objects:
        hit_info = obj.intersect(ray)
        if hit_info is not None:
            if hit is None or (hit_info[0] < hit).all():
                hit, hit_norm, hit_obj = hit_info[0], hit_info[1], obj

    # If no intersection, return background color
    if hit is None:
        return np.array([0, 0, 0])

    # Compute ambient and diffuse lighting
    color = np.array([0, 0, 0], dtype=np.float64)
    for light in lights:
        if light.position is not None:
            light_dir = normalize(light.position - hit)
            light_distance = norm(light.position - hit)
            # Check if the point is in shadow
            shadow_ray = Ray(hit + 1e-6 * hit_norm, light_dir)
            shadow_hit, _, shadow_obj = None, None, None
            for obj in objects:
                shadow_hit_info = obj.intersect(shadow_ray)
                if shadow_hit_info is not None:
                    if shadow_hit is None or (shadow_hit_info[0] < shadow_hit).any():
                        shadow_hit, _, shadow_obj = shadow_hit_info[0], shadow_hit_info[1], obj
            if shadow_hit is not None and (shadow_hit < light_distance).all():
                # Point is in shadow
                continue
            # Compute diffuse shading
            diffuse = max(np.dot(hit_norm, light_dir), 0)
            color += light.color * hit_obj.color * diffuse

            # Compute specular shading
            reflection_dir = reflect(-light_dir, hit_norm)
            specular = np.dot(ray.direction, reflection_dir)
            if specular > 0:
                specular = pow(specular, 32)
                color += light.color * specular

    return color


# Define scene geometry (e.g., spheres, planes, lights)
sphere1 = Sphere(np.array([1.0, 0.0, 5.0]), 1.0, np.array([0.0, 1.0, 0.0]))
sphere2 = Sphere(np.array([-1.0, 0.0, 4.0]), 1.0, np.array([1.0, 0.0, 0.0]))
plane1 = Plane(np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.5, 0.5, 0.5]))
lights = [Light(np.array([-2.0, 2.0, 0.0]), np.array([1.0, 1.0, 1.0]))]

# Define camera position, orientation, and field of view
camera_pos = np.array([0.0, 0.0, 0.0])
camera_dir = np.array([0.0, 0.0, 1.0])
camera_fov = np.pi / 3.0

# Define image size
image_width = 640
image_height = 480

# Compute image plane size
aspect_ratio = float(image_width) / float(image_height)
view_width = 2.0 * np.tan(camera_fov / 2.0)
view_height = view_width / aspect_ratio

# Generate image
image = np.zeros((image_height, image_width, 3), dtype=np.float64)
for y in range(image_height):
    for x in range(image_width):
        # Compute the ray direction for this pixel
        view_x = (float(x) / float(image_width - 1) - 0.5) * view_width
        view_y = (float(y) / float(image_height - 1) - 0.5) * view_height
        ray_dir = normalize(np.array([view_x, view_y, 1.0]))

        # Cast the ray and get the color
        ray = Ray(camera_pos, ray_dir)
        color = cast_ray(ray, [sphere1, sphere2, plane1], lights)

        # Set the pixel color in the image
        image[y, x] = color

# Save the image as a PNG file
image = np.clip(image, 0.0, 1.0) * 255.0
image = image.astype(np.uint8)
print(image)
Image.fromarray(image).save("output50.png")

end_time = time()
print(end_time-start_time)
