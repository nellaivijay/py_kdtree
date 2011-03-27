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
import array

cdef class BestPair:
  """Represents distance to a node from another node."""
  cdef int node_num
  cdef double dist

  def __cinit__(self, int node_num, double dist):
    self.node_num = node_num
    self.dist = dist

  property node_num:
    def __get__(self):
      return self.node_num

  property dist:
    def __get__(self):
      return self.dist

# Note that taking the extra effort to make this a cdef class makes a 5x difference
# in runtime speed on 100k rows
cdef class PointData:
  """Representation of point data.  Exists so we can sort without lambda key, 
  which Cython does not support."""
  cdef int num
  cdef size_t curr_axis
  cdef object coords
  cdef size_t dims

  def __cinit__(self, int num, coords):
    self.num = num
    self.coords = array.array('d', coords)
    self.curr_axis = 0
    self.dims = len(coords)

  def __cmp__(self, PointData other):
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
  cdef PointData data
  cdef KDTreeNode left
  cdef KDTreeNode right

  def __init__(self, int number, coords):
    """Initializer to pass in the number of the node and its coordinates"""
    self.data = PointData(number, array.array('d', coords))
    self.left = None
    self.right = None

  property num:
    def __get__(self):
      return self.data.num

  property left:
    def __get__(self):
      return self.left

  property right:
    def __get__(self):
      return self.right

  cpdef run_nn_search(self, PointData search, size_t num_neighbors):
    """Runs a nearest neighbor search on the given point, which is defined
    by the PointData search param."""
    
    search_coords = search.coords
    cdef size_t search_len = len(search_coords)

    nearest = [None] * num_neighbors

    cdef size_t i
    nn_search(self, search, nearest, 0, num_neighbors, 0)
    output = []
    for i in xrange(num_neighbors):
      output.append(nearest[i].node_num)

    return output

cpdef print_preorder(KDTreeNode node):
  """Print the nodes of the tree in pre-order fashion."""
  if node is None:
    return
  print "%d [%f,%f]" % (node.num, node.data.coords[0], node.data.coords[1])
  print_preorder(node.left)
  print_preorder(node.right)

cdef KDTreeNode fill_tree_r(pointList, size_t depth):
  """Constructs a KD-tree and returns the root node.  pointList is a list of PointData elements."""
  if not pointList:
    # TODO would using a null sentinel node make all of this run faster?
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
  node.left = fill_tree_r(pointList[0:median], next_depth)
  node.right = fill_tree_r(pointList[median + 1:], next_depth)
  return node

cpdef KDTreeNode fill_tree(pointList):
  """Constructs a KD-tree and returns the root node.  pointList is a list of PointData elements."""
  return fill_tree_r(pointList, 0)

cdef size_t pick_axis(size_t depth, size_t dims):
  """Choose the axis to use based on the current depth and the number of dimensions.
  This is useful when building the tree and search for nearest neighbors.
  @param [in] depth The current depth in the tree.
  @param [in] dims The number of dimensions in the points used for the tree.
  @return The axis to use.
  """
  return depth % dims

# Searches for nearest neighbor of point using node as the root.
cdef size_t nn_search(KDTreeNode node, PointData search, nearest, \
                   size_t best_count, size_t num_neighbors, size_t depth) except? -2:
  """Runs a nearest neighbor search on the given point, which is defined
  by the point number 'search_num' and search coordinates 'search'."""
  if node is None:
    return best_count

  cdef size_t dims = search.dims
  cdef size_t axis = pick_axis(depth, dims)
  cdef int node_num = node.num
  cdef int search_num = search.num

  ncoords = node.data.coords
  scoords = search.coords
  cdef double neighbor_coord = ncoords[axis]
  cdef double search_coord = scoords[axis]

  if node.left is None and node.right is None:
    if node_num != search_num:
      best_count = add_best(nearest, best_count, node_num, ncoords, scoords, dims, num_neighbors)
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
    best_count = add_best(nearest, best_count, node_num, ncoords, scoords, dims, num_neighbors)

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

cdef inline double sqdist(a, b, size_t dims) except? -2:
  """Calculates Euclidean distance between a and b"""

  cdef size_t k
  cdef double dist = 0.0
  cdef double diff
  for k in xrange(dims):
    diff = a[k] - b[k]
    dist = dist + (diff * diff)
  return dist

cdef size_t add_best(nearest, size_t best_count, int node_num, coords, \
                     search_coords, size_t dims, size_t num_neighbors) except? -2:
  """
  Add the node_num to the list of best nodes if it is closer to the searched-for 
  node than any current nodes.

  Returns: The count of elements in the BestPair list
  """
  # due to the constraints of the problem, we need to check each node
  # before assigning it as the final best choice to ensure it is not
  # equal to the searched-for point

  cdef double sd = sqdist(coords, search_coords, dims)

  cdef BestPair candidate = BestPair(node_num, sd)
  # search through linearly to maintain sorted order
  cdef size_t last_idx
  if best_count < num_neighbors:
    last_idx = best_count
  else:
    last_idx = best_count - 1

  cdef size_t idx, x
  cdef BestPair pair
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
