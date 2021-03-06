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
  struct point_data:
    int num
    double * coords
    size_t dims
    size_t curr_axis

  struct kdtree_node:
    point_data *data
    kdtree_node *left
    kdtree_node *right

  extern void c_run_nn_search "run_nn_search" (kdtree_node *, size_t, point_data, int[])
  extern kdtree_node * c_fill_tree "fill_tree" (point_data **, size_t)
  extern void free_tree(kdtree_node *)

cdef extern from "stdlib.h":
  void free(void* ptr)
  void* malloc(size_t size)

cdef class KDTreeNode:
  """A C extension class to the KDTree C code"""
  cdef kdtree_node *root

  def __dealloc__(self):
    """free the memory associated with root and its children"""
    if NULL != self.root:
      free_tree(self.root)
      # TODO let the C code handle this once we've fixed the NULL assignment problem
      self.root = NULL

  def __init__(self, pointList):
    cdef point_data **points
    cdef double * coords = NULL
    cdef size_t num_points, i, d, point_num
    cdef size_t dims = 0
    if NULL == self.root:
      num_points = len(pointList)
      points = <point_data **>malloc(num_points * sizeof(point_data *))
      if not points:
        raise MemoryError()

      try:
        # build up the point_data array that the C code desires
        for i in xrange(num_points):
          curr_point = pointList[i]
          point_num = curr_point[0]

          in_points = curr_point[1]
          dims = len(in_points)

          points[i] = <point_data *>malloc(sizeof(point_data))
          if not points[i]:
            raise MemoryError()

          points[i].num = point_num
          points[i].dims = dims;

          points[i].coords = <double *>malloc(dims * sizeof(double))
          if not points[i].coords:
            raise MemoryError()

          for d in xrange(dims):
            points[i].coords[d] = in_points[d]

        self.root = c_fill_tree(points, num_points)
      finally:
        for i in xrange(num_points):
          if NULL != points[i]:
            if NULL != points[i].coords:
              free(points[i].coords)
              points[i].coords = NULL
            free(points[i])
            points[i] = NULL
        free(points)
        points = NULL

  cpdef run_nn_search(self, int search_num, search, size_t num_neighbors):
    """Runs a nearest neighbor search on the given point, which is defined
    by the point number 'search_num' and search coordinates 'search'."""
    cdef size_t search_len = len(search)
    cdef point_data pd
    pd.dims = search_len
    pd.num = search_num

    pd.coords = <double *>malloc(search_len * sizeof(double))
    if not pd.coords:
      raise MemoryError()

    cdef size_t i
    for i in xrange(search_len):
      pd.coords[i] = search[i]

    cdef int *best = <int *>malloc(num_neighbors * sizeof(int))
    if not best:
      raise MemoryError()
    try:
      c_run_nn_search(self.root, num_neighbors, pd, best)
      output = []

      for i in xrange(num_neighbors):
        output.append(best[i])
      return output
    finally:
      if NULL != best:
        free(best)
      if NULL != pd.coords:
        free(pd.coords)
