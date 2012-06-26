
'''
Cython wrapper by Colin Lea
June 2012

Quickshift algorithm is part of vlfeat: http://www.vlfeat.org/
Copyright (C) 2007-12, Andrea Vedaldi and Brian Fulkerson
All rights reserved.

--Requirements--
Python 2.x (tested with 2.7)
Numpy 1.x (tested with 1.7)
Cython
VlFeat (tested with 0.9.14)

--Optional requirement-- 
Image module

--Compilation--
Run the following in the pyQuickshift folder:
python setup.py build_ext --inplace

--Example Usage--

import Image
import numpy as np
import pyQuickShift as qs

imgRaw = Image.open('/Users/colin/libs/vlfeat/data/a.jpg')
imgRGB = np.array(imgRaw, dtype=uint8)
img = np.ascontiguousarray(imgRaw, dtype=np.double)

if 0: # 3 channel
	labels, dists, density = qs.quickshift_3D(img*.5, 2, 20) # Image, kernel size, maxDistance
else: # 1 channel
	im2D = np.ascontiguousarray(img[:,:,2], dtype=double)
	labels, dists, density = qs.quickshift_2D(im2D*.5, 2, 20) # Image, kernel size, maxDistance

print "There are", len(unique(labels)), "superpixels"

#Paint the superpixels with their average color
if len(unique(labels)) < 10000:
	imgColor = np.empty_like(imgRGB)
	for l in unique(labels):
		imgColor[labels==l] = imgRGB[labels==l].mean(0)
	imshow(imgColor)

'''



import numpy as np
cimport numpy as np

np.import_array()
from libcpp cimport bool

ctypedef double vl_qs_type

cdef extern from "math.h":
	cdef int floor(double)

cdef extern from "../quickshift.h":
	cdef struct VlQS
	cdef VlQS* vl_quickshift_new (vl_qs_type*, int, int, int)
	cdef void  vl_quickshift_process (VlQS *q)
	cdef void  vl_quickshift_set_kernel_size (VlQS*, vl_qs_type)
	cdef void  vl_quickshift_set_max_dist (VlQS*, vl_qs_type)
	cdef vl_quickshift_set_medoid (VlQS*, vl_bool)
	cdef void  vl_quickshift_delete(VlQS*)
	cdef int*  vl_quickshift_get_parents(VlQS*)
	cdef vl_qs_type* vl_quickshift_get_dists(VlQS*)
	cdef vl_qs_type* vl_quickshift_get_density(VlQS*)

	cdef vl_qs_type vl_quickshift_get_max_dist(VlQS*)
	cdef vl_qs_type vl_quickshift_get_kernel_size(VlQS*)


def quickshift_2D(np.ndarray[np.double_t, ndim=2] im, double kernelSize=2, double maxDist=-1):#, bool medoid=False):
	cdef int height = im.shape[0]
	cdef int width = im.shape[1]
	cdef int channels = 1
	cdef int old, new

	cdef VlQS* obj
	cdef int* parents
	cdef vl_qs_type* distances
	cdef vl_qs_type* density_

	obj = vl_quickshift_new(<double*>im.data, height, width, channels)

	# Set kernel size, max distance
	vl_quickshift_set_kernel_size (obj, kernelSize)
	if maxDist > 0:
		vl_quickshift_set_max_dist(obj, maxDist)

	# Process
	vl_quickshift_process(obj)
	# Get data
	parents = vl_quickshift_get_parents(obj)
	distances = vl_quickshift_get_dists(obj)
	density_ = vl_quickshift_get_density(obj)

	# For each pixel we must follow the trail to get to the topmost node Get base label
	for i in range(height*width):
		while 1:
			old = parents[i]
			new = parents[old]
			# If at uppermost node, break
			if new == old:
				break
			else:
				parents[i] = new

	cdef np.npy_intp shape[2]
	shape[0] = height
	shape[1] = width
	npLabels = np.PyArray_SimpleNewFromData(2, shape, np.NPY_INT32, <void*>parents)
	dists = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, <void*>distances)
	density = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, <void*>density_)

	# vl_quickshift_delete (obj)

	return npLabels, dists, density



def quickshift_3D(np.ndarray[np.double_t, ndim=3] im, double kernelSize=2, double maxDist=-1):#, bool medoid=False):
	cdef int height = im.shape[0]
	cdef int width = im.shape[1]
	cdef int channels = 3
	cdef int old, new

	cdef VlQS* obj
	cdef int* parents
	cdef vl_qs_type* distances
	cdef vl_qs_type* density_

	obj = vl_quickshift_new(<double*>im.data, height, width, channels)

	# Set kernel size, max distance
	vl_quickshift_set_kernel_size (obj, kernelSize)
	if maxDist > 0:
		vl_quickshift_set_max_dist(obj, maxDist)

	# Process
	vl_quickshift_process(obj)
	# Get data
	parents = vl_quickshift_get_parents(obj)
	distances = vl_quickshift_get_dists(obj)
	density_ = vl_quickshift_get_density(obj)

	# For each pixel we must follow the trail to get to the topmost node Get base label
	for i in range(height*width):
		while 1:
			old = parents[i]
			new = parents[old]
			# If at uppermost node, break
			if new == old:
				break
			else:
				parents[i] = new

	cdef np.npy_intp shape[2]
	shape[0] = height
	shape[1] = width
	npLabels = np.PyArray_SimpleNewFromData(2, shape, np.NPY_INT32, <void*>parents)
	dists = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, <void*>distances)
	density = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, <void*>density_)

	# vl_quickshift_delete (obj)

	return npLabels, dists, density
