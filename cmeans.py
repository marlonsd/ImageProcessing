#!/usr/bin/env python
# -*- coding: utf-8-

import math, sys
import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt
from skfuzzy.cluster import cmeans as fcmeans
from sklearn.cluster import KMeans as kmeans
from time import time

from skimage.color import rgb2gray

from collections import defaultdict

BLACK	= 0
RED 	= (256/5)
GREEN	= RED + RED
BLUE	= GREEN + RED
WHITE	= BLUE + RED

COLORS 	= np.array((BLACK, RED, GREEN, BLUE, WHITE))


def neighborhood(image):

	mean, std = [], []

	for i, line in enumerate(image):
		for j, point in enumerate(line):
			comp = []

			try:
				comp.append(image[i-1][j-1])
			except:
				pass

			try:
				comp.append(image[i+1][j-1])
			except:
				pass

			try:
				comp.append(image[i][j-1])
			except:
				pass

			try:
				comp.append(image[i-1][j])
			except:
				pass

			try:
				comp.append(image[i][j])
			except:
				pass

			try:
				comp.append(image[i+1][j])
			except:
				pass

			try:
				comp.append(image[i-1][j+1])
			except:
				pass

			try:
				comp.append(image[i][j+1])
			except:
				pass


			try:
				comp.append(image[i+1][j+1])
			except:
				pass


			mean.append(np.mean(comp))
			std.append(np.std(comp))

	return mean, std


#	Function that generates n diferent colors
def coloring(n):

	def sameColor(color, set_colors):

		if not set_color:
			return False

		for c in set_colors:
			if (color == c):
				return True
 
	r = lambda: np.random.randint(0,256)

	set_color = []

	while len(set_color) < n:
		color = (r(), r(), r())

		if not sameColor(color, set_color):
			set_color.append(color)

	return set_color

#	Pre-processing of the image, in order to use them in the clustering algorithms
def image_processing(image):
	input_image = []
	for line in image:
	    for column in line:
	        input_image.append(column)

	# mean, std = neighborhood(image)
	# new_image = [input_image, mean, std]
	# return new_image

	# return [input_image]

def fuzzy_cmeans(image, n_clusters, m, error, maxiter):

	"""
	Clustering
		cntr : 2d array, size (S, c) 
			Cluster centers. Data for each center along each feature provided
			for every cluster (of the `c` requested clusters).
		U : 2d array, (S, N)
			Final fuzzy c-partitioned matrix.
		U0 : 2d array, (S, N)
			Initial guess at fuzzy c-partitioned matrix (either provided U_init or
			random guess used if U_init was not provided).
		d : 2d array, (S, N)
			Final Euclidian distance matrix.
		Jm : 1d array, length P
			Objective function history.
		p : int
			Number of iterations run.
		fpc : float
			Final fuzzy partition coefficient.
	"""

	input_image = np.array(image_processing(image))
	# input_image = np.reshape(input_image, (input_image.shape[0], 3))

	cntr, U, U0, d, Jm, p, fpc = fcmeans(data=input_image, c=n_clusters, m=m, error=error, maxiter=maxiter)

	coeficient = U.T

	comp = np.argsort(cntr[:,0], axis=0)
	indexes = range(len(comp))

	for pos in range(len(comp)):
		indexes[comp[pos]] = pos
	indexes = np.array(indexes)


	print cntr

	set_color = COLORS[indexes].tolist()

	p = 0
	image_f = image
	for pos_l, line in enumerate(image_f):
		for pos_p, point in enumerate(line):
			image_f[pos_l][pos_p] = set_color[np.argmax(coeficient[p])]
			p+=1

	return image_f

def cmeans(image, n_clusters):

	input_image = np.array(image_processing(image))
	# input_image = np.reshape(input_image, (input_image.shape[0], 1))

	classifier = kmeans(n_clusters=n_clusters)
	image_k = image

	new_image = classifier.fit_predict(input_image.T)

	comp = np.argsort(classifier.cluster_centers_[:,0], axis=0)
	indexes = range(len(comp))

	for pos in range(len(comp)):
		indexes[comp[pos]] = pos
	indexes = np.array(indexes)


	set_color = COLORS[indexes].tolist()

	# set_color = classifier.cluster_centers_

	print classifier.cluster_centers_

	p = 0
	for pos_l, line in enumerate(image):
		for pos_p, point in enumerate(line):
				image_k[pos_l][pos_p] = set_color[new_image[p]]
				p+=1

	return image_k


# execution = [	defaultdict(method='fcmeans', cluster=3, m= 2.0, error= 1e-5, maxiter=1000),
# 				defaultdict(method='fcmeans', cluster=3, m= 2.5, error= 1e-5, maxiter=1000),
# 				defaultdict(method='fcmeans', cluster=3, m= 3.0, error= 1e-5, maxiter=1000),
# 				defaultdict(method='fcmeans', cluster=3, m= 2.0, error= 1e-10, maxiter=1000),
# 				defaultdict(method='fcmeans', cluster=3, m= 2.5, error= 1e-10, maxiter=1000),
# 				defaultdict(method='fcmeans', cluster=3, m= 3.0, error= 1e-10, maxiter=1000),

# 				defaultdict(method='fcmeans', cluster=4, m= 2.0, error= 1e-5, maxiter=1000),
# 				defaultdict(method='fcmeans', cluster=4, m= 2.5, error= 1e-5, maxiter=1000),
# 				defaultdict(method='fcmeans', cluster=4, m= 3.0, error= 1e-5, maxiter=1000),
# 				defaultdict(method='fcmeans', cluster=4, m= 2.0, error= 1e-10, maxiter=1000),
# 				defaultdict(method='fcmeans', cluster=4, m= 2.5, error= 1e-10, maxiter=1000),
# 				defaultdict(method='fcmeans', cluster=4, m= 3.0, error= 1e-10, maxiter=1000),

# 				defaultdict(method='fcmeans', cluster=5, m= 2.0, error= 1e-5, maxiter=1000),
# 				defaultdict(method='fcmeans', cluster=5, m= 2.5, error= 1e-5, maxiter=1000),
# 				defaultdict(method='fcmeans', cluster=5, m= 3.0, error= 1e-5, maxiter=1000),
# 				defaultdict(method='fcmeans', cluster=5, m= 2.0, error= 1e-10, maxiter=1000),
# 				defaultdict(method='fcmeans', cluster=5, m= 2.5, error= 1e-10, maxiter=1000),
# 				defaultdict(method='fcmeans', cluster=5, m= 3.0, error= 1e-10, maxiter=1000),
				
# 				]

n_clusters = 3
metodo = 'cmeans'
m, error, maxiter = 2.0, 1e-5, 100

if len(sys.argv) > 1:

	if not (sys.argv[1].lower() == 'cmeans' or sys.argv[1].lower() == 'fcmeans'):
		print 'ERRO: A escolha do método deve ser feita usando cmeans ou fcmenas'
		print
		sys.exit(1)

	metodo = sys.argv[1]

if len(sys.argv) > 2:

	try:
		n_clusters = int (sys.argv[2])
	except:
		print 'ERRO: Número de clusters deve ser um valor numérico'
		print
		sys.exit(1)

# m, error, maxiter
if len(sys.argv) > 5:
	m, error, maxiter = float (sys.argv[3]), float (sys.argv[4]), int (sys.argv[5])

execution = [defaultdict(method=metodo, cluster=n_clusters, m=m, error=error, maxiter=maxiter)]

t0 = time()

image = io.imread('BOM_ECOLI.png')

print 'loading data', time() - t0, 's'
print

i = 0
for program in execution:

	result = []

	method = program['method']
	n_clusters = program['cluster']
	m = program['m']
	error = program['error']
	maxiter = program['maxiter']

	t0 = time()

	if method == 'cmeans':
		clustered_image = cmeans(image, n_clusters)
	else:
		clustered_image = fuzzy_cmeans(image, n_clusters, m, error, maxiter)

	t1 = time()

	filename = 'out_'+method+'_'+str(n_clusters)+'clusters'+'_m'+str(m)+'_error'+str(error)+'_'+str(i)
	io.imsave(filename+'.png', clustered_image)

	print filename, (t1 - t0), 's'
	print

	i+=1

	# results.append([filename,float(time() - t0)])

# for result in results:
# 	print result[0], ',' ,result[1]