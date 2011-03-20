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

# Original kdtree and knn_search code were combinations of code at
# http://code.google.com/p/python-kdtree/
# and
# http://en.wikipedia.org/wiki/Kd-tree
cdef extern from "stdlib.h":
  void free(void* ptr)
  void* malloc(size_t size)

cdef struct best_pair:
  int node_num
  double dist

cdef struct point_data:
  int num
  double *coords
  size_t dims
  size_t curr_axis

# Note that taking the extra effort to make this a cdef class makes a 5x difference
# in runtime speed on 100k rows
cdef class PointData:
  """Representation of point data.  Exists so we can sort without lambda key, 
  which Cython does not support."""
  cdef int num
  cdef size_t curr_axis
  cdef object coords
  cdef size_t dims

  def __cinit__(self, num, coords):
    self.num = num
    self.coords = coords
    self.curr_axis = 0
    self.dims = len(coords)

  def __cmp__(self, other):
    if other is None:
      return -1
    axis = self.curr_axis
    cdef double s_coord = self.coords[axis]
    cdef double o_coord = other.coords[axis]
    if s_coord < o_coord:
      return -1
    if s_coord > o_coord:
      return 1
    return 0

  def __str__(self):
    return "%d, %s" % (self.num, self.coords)

  def __repr__(self):
    return "%d, %s" % (self.num, self.coords)

  property dims:
    def __get__(self):
      return self.dims

  property num:
    def __get__(self):
      return self.num
    def __set__(self, int value):
      self.num = value

  property curr_axis:
    def __get__(self):
      return self.curr_axis
    def __set__(self, size_t value):
      self.curr_axis = value

  property coords:
    def __get__(self):
      return self.coords
    def __set__(self, object value):
      self.coords = value

cdef class KDTreeNode:
  """A C extension class to the KDTree C code"""
  cdef point_data *data
  cdef KDTreeNode left
  cdef KDTreeNode right

  def __cinit__(self, int number, coords):
    """Initializer to pass in the number of the node and its coordinates"""
    cdef size_t dims = len(coords)
    self.data = <point_data *>malloc(sizeof(point_data))
    if not self.data:
      raise MemoryError()
    self.data.coords = <double *>malloc(dims * sizeof(double))
    if not self.data.coords:
      raise MemoryError()
    cdef size_t i
    for i in xrange(dims):
      self.data.coords[i] = coords[i]
    self.data.num = number
    self.data.dims = dims
    self.left = None
    self.right = None

  def __dealloc__(self):
    """free the memory associated with this tree"""
    if NULL != self.data:
      if NULL != self.data.coords:
        free(self.data.coords)
        self.data.coords = NULL
      free(self.data)
      self.data = NULL

  property num:
    def __get__(self):
      return self.data.num

  property left:
    def __get__(self):
      return self.left

  property right:
    def __get__(self):
      return self.right

  cpdef run_nn_search(self, search, size_t num_neighbors):
    """Runs a nearest neighbor search on the given point, which is defined
    by the PointData search param."""
    
    search_coords = search.coords
    cdef size_t search_len = len(search_coords)
    cdef point_data pd = mk_point_data(search_coords, search.num, search_len)

    # cdef best_pair[num_neighbors] won't work; Cython complains about wanting 
    # constants for allocation, so we need to dynamically allocate it
    cdef best_pair *nearest = <best_pair *>malloc(num_neighbors * sizeof(best_pair))
    if not nearest:
      raise MemoryError()

    cdef size_t i
    try:
      nn_search(self, pd, nearest, 0, num_neighbors, 0)
      output = []
      for i in xrange(num_neighbors):
        output.append(nearest[i].node_num)

      return output
    finally:
      if NULL != nearest:
        free(nearest)
        nearest = NULL
      if NULL != pd.coords:
        free(pd.coords)
        pd.coords = NULL

cdef point_data mk_point_data(coords, size_t search_num, size_t search_sz):
  """Creates a point_data structure from the input.  This allocates memory for 
  the coords field of the point_data structure, so don't forget to free it."""
  cdef point_data pd
  pd.dims = search_sz
  pd.num = search_num

  pd.coords = <double *>malloc(search_sz * sizeof(double))
  if not pd.coords:
    raise MemoryError()

  cdef size_t i
  for i in xrange(search_sz):
    pd.coords[i] = coords[i]
  return pd

cpdef print_preorder(KDTreeNode node):
  """Print the nodes of the tree in pre-order fashion."""
  if node is None:
    return
  print "%d [%f,%f]" % (node.num, node.data.coords[0], node.data.coords[1])
  print_preorder(node.left)
  print_preorder(node.right)

cpdef KDTreeNode fill_tree(pointList, size_t depth=0):
  """Constructs a KD-tree and returns the root node.  pointList is a list of PointData elements."""
  if not pointList:
    return None

  cdef size_t dims = pointList[0].dims # assumes all points have the same dimension
  cdef size_t axis = pick_axis(depth, dims)
  cdef PointData p
  for p in pointList:
    p.curr_axis = axis

  # Sort point list and choose median as pivot element
  pointList.sort()

  cdef size_t median = len(pointList) / 2

  pl_median = pointList[median]
  cdef int next_depth = depth + 1
  cdef KDTreeNode node = KDTreeNode(pl_median.num, pl_median.coords)
  node.left = fill_tree(pointList[0:median], next_depth)
  node.right = fill_tree(pointList[median + 1:], next_depth)
  return node

cdef size_t pick_axis(size_t depth, size_t dims):
  """Choose the axis to use based on the current depth and the number of dimensions.
  This is useful when building the tree and search for nearest neighbors.
  @param [in] depth The current depth in the tree.
  @param [in] dims The number of dimensions in the points used for the tree.
  @return The axis to use.
  """
  return depth % dims

# Searches for nearest neighbor of point using node as the root.
cdef size_t nn_search(KDTreeNode node, point_data search, best_pair nearest[], \
                   size_t best_count, size_t num_neighbors, size_t depth) except? -2:
  """Runs a nearest neighbor search on the given point, which is defined
  by the point number 'search_num' and search coordinates 'search'."""
  if node is None:
    return best_count

  cdef size_t dims = search.dims
  cdef size_t axis = pick_axis(depth, dims)
  cdef int node_num = node.num
  cdef int search_num = search.num

  cdef double neighbor_coord = node.data.coords[axis]
  cdef double search_coord = search.coords[axis]

  if node.left is None and node.right is None:
    if node_num != search_num:
      best_count = add_best(nearest, best_count, node, search, num_neighbors)
    return best_count


  # compare query point and current node along the axis to see which tree is
  # far and which is near
  if search_coord < neighbor_coord:
    near = node.left
    far = node.right
  else:
    near = node.right
    far = node.left

  cdef size_t next_depth = depth + 1
  # search the near branch
  if near is not None:
    best_count = nn_search(near, search, nearest, best_count, num_neighbors, next_depth)

  # If the current node is closer overall than the current best
  if node_num != search_num:
    best_count = add_best(nearest, best_count, node, search, num_neighbors)

  cdef double largest = -1.0
  if best_count > 0:
    largest = nearest[best_count - 1].dist

  cdef double diff
  cdef double sq_diff
  # maybe search the away branch
  cdef size_t search_other
  if far is not None:
    search_other = 0
    if largest < 0:
      search_other = 1
    else:
      diff = neighbor_coord - search_coord
      sq_diff = diff * diff
      if sq_diff < largest:
        search_other = 1
    if 1 == search_other:
      best_count = nn_search(far, search, nearest, best_count, num_neighbors, next_depth)
  return best_count

cdef inline double sqdist(double a[], double b[], size_t dims) except? -2:
  """Calculates Euclidean distance between a and b"""

  cdef size_t k
  cdef double dist = 0.0
  cdef double diff
  for k in xrange(dims):
    diff = a[k] - b[k]
    dist = dist + (diff * diff)
  return dist

cdef size_t add_best(best_pair nearest[], size_t best_count, KDTreeNode node, \
                     point_data search, size_t num_neighbors) except? -2:
  """
  Add the node_num to the list of best nodes if it is closer to the searched-for 
  node than any current nodes.

  Returns: The count of elements in the best_pair list
  """
  # due to the constraints of the problem, we need to check each node
  # before assigning it as the final best choice to ensure it is not
  # equal to the searched-for point

  cdef double sd = sqdist(node.data.coords, search.coords, search.dims)

  cdef best_pair candidate = best_pair(node.num, sd)
  # search through linearly to maintain sorted order
  cdef size_t last_idx
  if best_count < num_neighbors:
    last_idx = best_count
  else:
    last_idx = best_count - 1

  cdef size_t idx, x
  cdef best_pair pair
  for idx in xrange(best_count):
    pair = nearest[idx]
    if pair.dist > sd:
      # push elements down
      for x in xrange(last_idx, idx, -1):
        nearest[x] = nearest[x - 1]
      nearest[idx] = candidate
      return last_idx + 1
  if best_count < num_neighbors:
    # didn't find an insert spot, so insert at the end
    nearest[best_count] = candidate
    return best_count + 1
  return best_count
