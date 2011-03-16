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

cdef int LIMIT
LIMIT = 3

cdef class BestPair:
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

cdef class KDTreeNode:
  """A C extension class to the KDTree C code"""
  cdef int number
  cdef double coords[2]
  cdef KDTreeNode left
  cdef KDTreeNode right

  def __cinit__(self, int number, double xcoord, double ycoord):
    self.coords[0] = xcoord
    self.coords[1] = ycoord
    self.number = number
    self.left = None
    self.right = None

  property number:
    def __get__(self):
      return self.number

  property left:
    def __get__(self):
      return self.left

  property right:
    def __get__(self):
      return self.right

  cpdef run_nn_search(self, int search_num, search):
    """Runs a nearest neighbor search on the given point, which is defined
    by the point number 'search_num' and search coordinates 'search'."""
    best = [None] * 3
    cdef double search_arr[2]
    search_arr[0] = search[0]
    search_arr[1] = search[1]
    nn_search(self, search_num, search_arr, best, 0, 0)
    output = []
    for x in best:
      output.append(x.node_num)
    return output

def axis0PointKey(point):
  """Sort for axis zero (i.e. X axis)"""
  return point[1][0]

def axis1PointKey(point):
  """Sort for axis one (i.e. Y axis)"""
  return point[1][1]

cpdef print_preorder(KDTreeNode node):
  """Print the nodes of the tree in pre-order fashion."""
  if node is None:
    return
  print "%d [%f,%f]" % (node.number, node.coords[0], node.coords[1])
  print_preorder(node.left)
  print_preorder(node.right)

cpdef KDTreeNode kdtree(pointList, int axis):
  """Constructs a KD-tree and returns the root node.  pointList is a tuple of (ID, [x,y])"""
  if not pointList:
    return None

  # Normally we'd elect axis based on depth so that axis cycles through 
  # all valid values.  Here we know we have 2-dimensional values, so don't 
  # waste time calculating, but just alternate
  #k = len(pointList[0][1]) # assumes all points have the same dimension

  # Sort point list and choose median as pivot element
  # Note that pointList[0] = number, pointList[1] = lat, long coordinates
  #pointList.sort(key=lambda point: point[1][axis])
  if axis == 0:
    pointList.sort(key=axis0PointKey)
  else:
    pointList.sort(key=axis1PointKey)

  cdef int median = len(pointList) / 2

  cdef int next_axis = 1
  if 1 == axis:
    next_axis = 0
  cdef KDTreeNode node
  node = KDTreeNode(pointList[median][0], pointList[median][1][0], pointList[median][1][1])
  cdef KDTreeNode left = kdtree(pointList[0:median], next_axis)
  cdef KDTreeNode right = kdtree(pointList[median + 1:], next_axis)
  node.left = left
  node.right = right
  return node


# Searches for nearest neighbor of point using node as the root.
cdef int nn_search(KDTreeNode node, int search_num, double search[], best, int count, int axis) except? -2:
  """Runs a nearest neighbor search on the given point, which is defined
  by the point number 'search_num' and search coordinates 'search'."""
  if node is None:
    return count

  cdef int node_num = node.number
  cdef double nodepoint[2]
  nodepoint[0] = node.coords[0]
  nodepoint[1] = node.coords[1]

  if node.left is None and node.right is None:
    if node_num != search_num:
      count = add_best(count, node_num, nodepoint, search, best)
    return count

  # Normally we'd elect axis based on depth so that axis cycles through 
  # all valid values.  Here we know we have 2-dimensional values, so don't 
  # waste time calculating, but just alternate
  #axis = depth % len(point)

  # compare query point and current node along the axis to see which tree is
  # far and which is near

  if search[axis] < nodepoint[axis]:
    near = node.left
    far = node.right
  else:
    near = node.right
    far = node.left

  # search the near branch
  cdef int next_axis = 1
  if 1 == axis:
    next_axis = 0
  if near is not None:
    count = nn_search(near, search_num, search, best, count, next_axis)

  # If the current node is closer overall than the current best
  if node_num != search_num:
    count = add_best(count, node_num, nodepoint, search, best)

  cdef double largest
  cdef BestPair bp
  if count > 0:
    bp = best[count - 1]
    largest = bp.dist
  else:
    largest = -1.0

  cdef double diff
  cdef double sq_diff
  # maybe search the away branch
  cdef int search_other
  if far is not None:
    search_other = 0
    if largest < 0:
      search_other = 1
    else:
      diff = nodepoint[axis] - search[axis]
      sq_diff = diff * diff
      if sq_diff < largest:
        search_other = 1
    if 1 == search_other:
      count = nn_search(far, search_num, search, best, count, next_axis)
  return count

cdef inline double sqdist(double a[], double b[]) except? -2:
  """Calculates Euclidean distance between a and b"""
  cdef double diffx = a[0] - b[0]
  cdef double diffy = a[1] - b[1]
  return (diffx * diffx) + (diffy * diffy)

cdef int add_best(int count, int node_num, double nodepoint[], double search[], best) except? -2:
  """
  Add the node_num to the list of best nodes if it is closer to the searched-for 
  node than any current nodes.

  Returns: The count of elements in the BestPair list
  """
  # due to the constraints of the problem, we need to check each node
  # before assigning it as the final best choice to ensure it is not
  # equal to the searched-for point

  cdef double sd = sqdist(nodepoint, search)

  cdef BestPair candidate = BestPair(node_num, sd)
  # search through linearly to maintain sorted order
  cdef int last_idx
  if count < LIMIT:
    last_idx = count
  else:
    last_idx = count - 1

  cdef int idx
  cdef int x
  cdef BestPair pair
  for idx in xrange(count):
    pair = best[idx]
    if pair.dist > sd:
      # push elements down
      # Use a while loop else Cython won't turn this into C
      #for x in xrange(last_idx, idx, -1):
      x = last_idx
      while x > idx:
        best[x] = best[x - 1]
        x = x - 1
      best[idx] = candidate
      return last_idx + 1
  if count < LIMIT:
    # didn't find an insert spot, so insert at the end
    best[count] = candidate
    return count + 1
  return count
