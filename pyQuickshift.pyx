
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
img = Image.open('/Users/colin/libs/vlfeat/data/a.jpg')

# Convert to a normalized double
img = np.array(img, dtype = double)
# img /= img.max()
img += np.random.rand(img.shape[0], img.shape[1], 3)/1000
# img = (img - img.min - (img.max()-img.min())
# img /= img.max()
labels, dists, density = qs.quickshift_3D(img, 2, 20, True)
print "There are", len(unique(labels)), "superpixels"

imgColor = np.empty_like(img)
for l in unique(labels):
	imgColor[labels==l] = img[labels==l].mean(0)
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


def quickshift_2D(np.ndarray[np.double_t, ndim=2] im, double kernelSize=5, double maxDist=-1):#, bool medoid=False):
	cdef int height = im.shape[0]
	cdef int width = im.shape[1]
	cdef int channels = 1

	cdef VlQS* obj = vl_quickshift_new(<double*>im.data, height, width, channels)

	# Set kernel size, max distance, and if medoid enabled
	vl_quickshift_set_kernel_size (obj, kernelSize)
	if maxDist > 0:
		vl_quickshift_set_max_dist(obj, maxDist)

	# if medoid == True:
	# 	vl_quickshift_set_medoid(obj, True)
	# else:
	# 	vl_quickshift_set_medoid(obj, False)
	
	# Process
	vl_quickshift_process(obj)
	# Get data
	cdef int* parents = vl_quickshift_get_parents(obj)
	cdef vl_qs_type* distances = vl_quickshift_get_dists(obj)
	cdef vl_qs_type* density_ = vl_quickshift_get_density(obj)

	# For each pixel we must follow the trail to get to the topmost node Get base label
	cdef int i2
	for i in range(height*width):
		while 1:
			i2 = parents[i]
			new = parents[i2]
			# If at uppermost node, break
			if new == i2:
				break
			else:
				parents[i] = new


	npLabels = np.PyArray_SimpleNewFromData(2, [height,width], np.NPY_UINT32, <void*>parents)
	dists = np.PyArray_SimpleNewFromData(2, [height,width], np.NPY_DOUBLE, <void*>distances)
	density = np.PyArray_SimpleNewFromData(2, [height,width], np.NPY_DOUBLE, <void*>density_)

	vl_quickshift_delete (obj)

	return npLabels, dists, density



def quickshift_3D(np.ndarray[np.double_t, ndim=3] im, double kernelSize=5, double maxDist=-1):#, bool medoid=False):
	cdef int height = im.shape[0]
	cdef int width = im.shape[1]
	cdef int channels = 3



	cdef VlQS* obj = vl_quickshift_new(<double*>im.data, height, width, channels)

	# Set kernel size, max distance, and if medoid enabled
	vl_quickshift_set_kernel_size (obj, kernelSize)
	if maxDist > 0:
		vl_quickshift_set_max_dist(obj, maxDist)
	# if medoid == 1:
	# 	vl_quickshift_set_medoid(obj, True)
	# else:
	# 	vl_quickshift_set_medoid(obj, False)

	# Process
	vl_quickshift_process(obj)
	# Get data
	cdef int* parents = vl_quickshift_get_parents(obj)
	cdef vl_qs_type* distances = vl_quickshift_get_dists(obj)
	cdef vl_qs_type* density_ = vl_quickshift_get_density(obj)

	# For each pixel we must follow the trail to get to the topmost node Get base label
	cdef int i2
	for i in range(height*width):
		while 1:
			i2 = parents[i]
			new = parents[i2]
			# If at uppermost node, break
			if new == i2:
				break
			else:
				parents[i] = new

	npLabels = np.PyArray_SimpleNewFromData(2, [height,width], np.NPY_UINT32, <void*>parents)
	dists = np.PyArray_SimpleNewFromData(2, [height,width], np.NPY_DOUBLE, <void*>distances)
	density = np.PyArray_SimpleNewFromData(2, [height,width], np.NPY_DOUBLE, <void*>density_)

	vl_quickshift_delete (obj)

	return npLabels, dists, density
