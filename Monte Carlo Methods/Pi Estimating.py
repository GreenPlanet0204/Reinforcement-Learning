import numpy as np
import math
import random
import matplotlib.pyplot as plt

square_size = 1
points_inside_circle = 0
points_inside_square = 0
sample_size = 1000
arc = np.linspace(0, np.pi/2, 100)

def generate_points(size):
  x = random.random()*size
  y = random.random()*size
  return (x, y)

def is_in_circle(point, size):
  return math.sqrt(point[0]**2 + point[1]**2) <= size

def compute_pi(points_inside_circle, points_inside_square):
  return 4. * (points_inside_circle / points_inside_square)

plt.axes().set_aspect('equal')
plt.plot(1*np.cos(arc), 1*np.sin(arc))

for i in range(sample_size):
  point = generate_points(square_size)
  plt.plot(point[0], point[1], 'c.')
  points_inside_square += 1
  if is_in_circle(point, square_size):
    points_inside_circle += 1
print("Approximate value of pi is {}".format(compute_pi(points_inside_circle, points_inside_square)))