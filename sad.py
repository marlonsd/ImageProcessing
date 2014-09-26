'''
Implementation of Sum of Absolute Differences algorithm for two Images
Author: Marlon Dias
https://github.com/marlonsd
'''

#!/usr/bin/env python
# -*- coding: utf-8-

import math, sys
import numpy as np
import skimage.io as io

def sad(image1, image2):

	# Finding smallest image
	# Condition: One of the dimensions has to be the same
	if len(image1) == len(image2):
		# Imagens' height are the same, find the smallest width
		length = image1 if (len(image1[0]) <= len(image2[0])) else image2
	else:
		# Images' height are different, assume that width is equal.
		length = image1 if (len(image1) <= len(image2)) else image2

	accu = 0.

	# Accumulate the absolute differences of the images
	for i in range(len(length)):
		for j in range(len(length[i])):
			accu += np.absolute(float(image1[i][j]) - float(image2[i][j]))

	return accu


if len(sys.argv) < 3:
	print "Two files must be informed for the comparison"
	sys.exit()

files = [io.imread(sys.argv[1]), io.imread(sys.argv[2])]

print 'sad('+str(sys.argv[1])+', '+str(sys.argv[2])+') =',sad(files[0], files[1])