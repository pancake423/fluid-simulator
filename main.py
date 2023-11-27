'''
Fluid Simulator
William Jackson
Phys 325

11/26/2023


The goal of this project is to simulate the motion of a fluid in a closed rectangular container,
using the Navier-Stokes equations.

References:
https://hplgit.github.io/INF5620/doc/pub/main_ns.pdf
https://math.stackexchange.com/questions/1255935/is-the-laplacian-a-vector-or-a-scalar
https://byjus.com/maths/divergence-and-curl
https://en.wikipedia.org/wiki/Gradient


The Navier-Stokes equations are:

∂u/∂t + u * ∇u = -1/ρ * ∇p + v * ∇²u

and 

∇u = 0.

u(x, y, t) is the velocity field (ie. finds velocity vector at a given point in space and time)
p(x, y, t) is the pressure field (ie. finds the pressure at a given point in space and time)
v is the viscosity constant of the fluid..
ρ (rho) is the density constant of the fluid.


This is definitely the most complex mathematical equation I've ever tried to wrap my head around.
After hours of research, I was able to find a paper that presents a numerical integration scheme
that I can understand how to implement (after a few more hours of research).

https://hplgit.github.io/INF5620/doc/pub/main_ns.pdf

This scheme relies on slight compressiblility, ∇u != 0, and a tuning parameter c.

for the velocity and pressure at a given point in space:

u_(n+1) = u_n - Δt * (u_n * ∇)u_n - Δt / ρ * ∇p_n + Δt * v * ∇²u_n

p_(n+1) = p_n - Δt * c^2 * ∇*u_n
'''

import numpy as np
import pygame

SIM_WIDTH = 100 #simulation grid width in cells
SIM_HEIGHT = 50 #simulation grid height in cells

VISCOSITY = 1
DENSITY = 1
TUNING = 20 #tuning parameter C

velocity = np.zeros((SIM_WIDTH, SIM_HEIGHT, 2)) #initial condition for velocity
pressure = np.ones((SIM_WIDTH, SIM_HEIGHT)) #initial condition for pressure

#step the velocity at one point forward by one iteration.
def velocity_step(vel, div_vel, grad_pres, laplace_vel, delta_t):
	velocity_term = -1 * delta_t * div_vel * vel
	pressure_term = -1 * delta_t / DENSITY * grad_pres
	viscosity_term = delta_t * VISCOSITY * laplace_vel

	return np.add(velocity_term, np.add(pressure_term, viscosity_term))

#step the pressure at one point forward by one iteration.
def pressure_step(pres, div_vel, delta_t):
	pressure_change = -1 * delta_t * TUNING * TUNING * div_vel
	return np.add()

#numerically finds the gradient of a function, represented as a 2d array of values.
#for values out-of-bounds on the array, replace the array value with <boundary>
def gradient(func, boundary):
	'''
	the gradient is the vector <∂f/∂x, ∂f/∂y>
	to find the partial derivative of the cell at x, y
	with respect to x:

	(f(x + 1) - f(x-1)) / 2

	with respect to y:
	(f(y + 1) - f(y - 1)) / 2
	'''
	array = np.pad(func, 1, constant_values=boundary)
	dimensions = array.shape
	out = np.zeros((*func.shape, 2))
	for x in range(1, dimensions[0]-1):
		for y in range(1, dimensions[1]-1):
			out[x-1][y-1][0] = (array[x+1][y] - array[x-1][y]) / 2
			out[x-1][y-1][1] = (array[x][y+1] - array[x][y-1]) / 2

	return out

#calculate the divergence of a function, represented as a 2d array of vectors (3d array)
#boundary is a vector that replaces the array values when out-of-bounds.
def divergence(func, boundary):
	'''
	the divergence of a vector valued function is the scalar

	∂f0/∂x + ∂f1/∂y
	
	which can be approximated for a cell x, y as

	(f(x + 1, y)[0] - f(x - 1, y)[0]) / 2 + (f(x, y+1)[1] - f(x, y-1)[1]) / 2 
	'''
	array = np.pad(func, ((1, 1), (1, 1), (0, 0)), constant_values = boundary)
	out = np.zeros((func.shape[0], func.shape[1]))
	dimensions = array.shape
	for x in range(1, dimensions[0]-1):
		for y in range(1, dimensions[1]-1):
			out[x-1][y-1] = (
				array[x+1][y][0] - array[x-1][y][0] 
				+ array[x][y+1][1] - array[x][y-1][1]
			) / 2
	return out

def laplace(func, boundary):
	'''
	the laplace operator of a vector v is:

	∂²v/∂x² + ∂²v/∂y²

	which can be approximated as:

	(∂f/∂x(x+1, y) - ∂f/∂x(x-1, y)) / 2 + (∂f/∂y(x, y+1) - ∂f/∂y(x, y-1)) / 2

	which expands to:

	((f(x+2, y) - f(x, y)) / 2 - (f(x, y) - f(x-2, y)) / 2) / 2 + ...

	and simplifies to:

	(f(x+2, y) + f(x-2, y) + f(x, y+2) + f(x, y-2)) / 4 - f(x, y)
	'''
	array = np.pad(func, ((2, 2), (2, 2), (0, 0)), constant_values = boundary)
	out = np.zeros_like(func)
	dimensions = array.shape
	for x in range(1, dimensions[0]-3):
		for y in range(1, dimensions[1]-3):
			vec = np.add(array[x+2][y], array[x-2][y])
			vec = np.add(vec, array[x][y+2])
			vec = np.add(vec, array[x][y-2])
			vec /= 4
			vec = np.subtract(vec, array[x][y])

			out[x-2][y-2][0] = vec[0]
			out[x-2][y-2][1] = vec[1]
	return out

