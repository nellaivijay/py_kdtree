#!/usr/bin/python
# Copyright 2011 Chris M Bouzek
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the version 3 of the GNU Lesser General Public License
# as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

cdef extern from "kdtree_raw.h":
  struct kdtree_node:
    int number
    double *coords
    kdtree_node * left
    kdtree_node * right

  struct point_data:
    int number
    double * coords

  extern void c_run_nn_search "run_nn_search" (kdtree_node *, int [], int, int, double search[])

cdef extern from "stdlib.h":
  void free(void* ptr)
  void* malloc(size_t size)

def axis0PointKey(point):
  """Sort for axis zero (i.e. X axis)"""
  return point[1][0]

def axis1PointKey(point):
  """Sort for axis one (i.e. Y axis)"""
    return point[1][1]


cdef kdtree_node * fill_tree(pointList, int axis):
  """ Constructs a KD-tree and returns the root node.  pointList is a tuple of (ID, [x,y])"""
  if not pointList:
    return NULL

  # Normally we'd elect axis based on depth so that axis cycles through 
  # all valid values.  Here we know we have 2-dimensional values, so don't 
  # waste time calculating, but just alternate
  #k = len(pointList[0][1]) # assumes all points have the same dimension

  # Sort point list and choose median as pivot element
  # Note that pointList[0] = number, pointList[1] = lat, long coordinates
  # sort is faster using strings, so we use point[1]
  #pointList.sort(key=lambda point: point[1][axis])
  if axis == 0:
    pointList.sort(key=axis0PointKey)
  else:
    pointList.sort(key=axis1PointKey)

  cdef int median = len(pointList) / 2

  cdef int next_axis = 1
  if 1 == axis:
    next_axis = 0

  cdef kdtree_node *node = <kdtree_node *>malloc(sizeof(kdtree_node))

  cdef int number = pointList[median][0]
  node.number = number
  cdef double *coords
  plMedian = pointList[median][2]
  cdef int point_sz = len(plMedian)
  coords = <double *>malloc(point_sz * sizeof(double))

  cdef int i
  for i in xrange(point_sz):
    coords[i] = plMedian[i]
  node.coords = coords
  node.left = fill_tree(pointList[0:median], next_axis)
  node.right = fill_tree(pointList[median + 1:], next_axis)
  return node

cdef class KDTreeNode:
  """A C extension class to the KDTree C code"""
  cdef kdtree_node *root
  def __cinit__(self):
    """Malloc the root of the tree"""
    pass
    #<kdtree_node *>malloc(sizeof(kdtree_node))

  cdef dealloc_tree(self, kdtree_node * node):
    """free the memory associated with root and its children"""
    if NULL == node:
      return
    self.dealloc_tree(node.left)
    self.dealloc_tree(node.right)
    free(node.coords)
    free(node)
    node = NULL

  def __dealloc__(self):
    self.dealloc_tree(self.root)

  def __init__(self, pointList):
    if NULL == self.root:
      self.root = fill_tree(pointList, 0)

  def run_nn_search(self, point_num, search):
    """Runs a nearest neighbor search on the given point, which is defined
    by the point number 'point_num' and search coordinates 'search'."""
    cdef int number = point_num
    cdef double search_point[2]
    search_point[0] = search[0]
    search_point[1] = search[1]

    cdef int best_sz = 3
    cdef int best[3]
    cdef int i
    c_run_nn_search(self.root, best, best_sz, point_num, search_point)
    output = []

    for i in xrange(best_sz):
      output.append(best[i])
    return output
