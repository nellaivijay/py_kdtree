/*
 * Copyright 2011 Chris M Bouzek
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the version 3 of the GNU Lesser General Public License
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <stdio.h>
#include <string.h>
#include "kdtree_raw.h"

typedef struct best_pair {
	int node_num;
	double dist;
} best_pair;

/* Utility functions */

/* Determine the largest element in the best neighbors array*/
static double largest_dist(best_pair best[], int count) {
	double largest = -1.0;
	if (count > 0) {
		largest = best[count - 1].dist;
	}
	return largest;
}

/*
 * Choose the next axis to use based on the current axis.
 */
static int pick_axis(int axis) {
  /* search the near branch */
  int next_axis;
	if (1 == axis) {
		next_axis = 0;
	} else {
		next_axis = 1;
	}
	return next_axis;
}

static double sqdist(double a[], double b[]) {
	double diffx = a[0] - b[0];
	double diffy = a[1] - b[1];
	return (diffx * diffx) + (diffy * diffy);
}

kdtree_node * make_tree(
		point_data points[], 
		int num_points, 
		int axis) {

  /*node = kdtree_node(pointList[median][0], pointList[median][2][0], pointList[median][2][1])
  cdef kdtree_node left = kdtree(pointList[0:median], next_axis)
  cdef kdtree_node right = kdtree(pointList[median + 1:], next_axis)*/
	int median = num_points / 2;
	int right_sz = num_points - median;
  int next_axis = pick_axis(axis);
	kdtree_node* node = malloc(sizeof(kdtree_node));
	if (NULL == node) {
		/* TODO error code */
		return NULL;
	}

	node->number = points[median].number;
	node->coords[0] = points[median].coords[0];
	node->coords[1] = points[median].coords[1];

	{
		point_data left_arr[median];
		memcpy (left_arr, points, median * sizeof(point_data));
		node->left = make_tree(left_arr, next_axis, median);
	}

	{
		point_data right_arr[right_sz];
		memcpy (right_arr, &points[median + 1], right_sz * sizeof(point_data));
		node->right = make_tree(right_arr, next_axis, right_sz);
	}
	return node;
}

/* Functions directly related to KD-tree functionality */

/* Adds the search to the list of best nodes if it is closer than the current best
   list. */
static int add_best(
		int count, 
		double nodepoint[], 
		int node_num, 
		double search[], 
		best_pair best[],
		int best_sz) {

	double sd = sqdist(nodepoint, search);
	int last_idx;
	if (count < best_sz) {
		last_idx = count;
	} else {
		last_idx = count - 1;
	}

	int idx;
	best_pair candidate;
	candidate.node_num = node_num;
	candidate.dist = sd;

	best_pair pair;
	/* search through linearly to maintain sorted order */
	for (idx = 0; idx < count; idx++) {
		pair = best[idx];
		if (pair.dist > sd) {
			/* push elements down */
			int x;
			for (x = last_idx; x > idx; x--) {
				best[x] = best[x - 1];
			}
			best[idx] = candidate;
			return last_idx + 1;
		}
	}

	if (count < best_sz) {
		/* didn't find an insert spot, so insert at the end */
		best[last_idx] = candidate;
		return last_idx + 1;
	}
  return count;
}

/* Searches for nearest neighbor of search using node as the root. */
static int nn_search(
		kdtree_node *node, 
		int search_num,
		double search[],
		best_pair best[], 
		int best_sz,
		int count, 
		int axis) {
  if (NULL == node) {
    return count;
	}

	int cnt = count;
  double *nodepoint = node->coords;
	int node_num = node->number;

  /* due to the constraints of the problem, we need to check each node
     before assigning it as the final best choice to ensure it is not
     equal to the searched-for point, hence node_num != search_num */
	if (NULL == node->left && NULL == node->right) {
    if (node_num != search_num) {
      cnt = add_best(cnt, nodepoint, node_num, search, best, best_sz);
		}
    return cnt;
	}

  /* Normally we'd select axis based on depth so that axis cycles through 
     all valid values.  Here we know we have 2-dimensional values, so don't 
     waste time calculating, but just alternate
     axis = depth % len(search) */

  /* compare query point and current node along the axis to see which tree is
     far and which is near */

	kdtree_node *near;
	kdtree_node *far;
  if (search[axis] < nodepoint[axis]) {
    near = (kdtree_node *)node->left;
    far = (kdtree_node *)node->right;
	} else {
    near = (kdtree_node *)node->right;
    far = (kdtree_node *)node->left;
	}

  /* search the near branch */
  int next_axis = pick_axis(axis);

	if (NULL != near) {
	  cnt = nn_search(near, search_num, search, best, best_sz, cnt, next_axis);
	}

  /* If the current node is closer overall than the current best */
  if (node_num != search_num) {
    cnt = add_best(cnt, nodepoint, node_num, search, best, best_sz);
	}

  /* maybe search the away branch */
	if (NULL != far) {
		double largest = largest_dist(best, cnt);
		int search_other = 0;
		if (largest < 0) {
			search_other = 1;
		} else {
			double diff = nodepoint[axis] - search[axis];
			if ((diff * diff) < largest) {
				search_other = 1;
			}
		}
		if (1 == search_other) {
			cnt = nn_search(far, search_num, search, best, best_sz, cnt, next_axis);
		}
	}
  return cnt;
}

/* Initializes the nearest neighbor search point and starts the search.
 * call it like run_nn_search(kdtree_node, search_num, (search_x, search_y))
 */
extern void
run_nn_search(kdtree_node *root, int best_nums[], int best_sz, int search_num, double search[]) {
	best_pair best[best_sz];
	nn_search(root, search_num, search, best, best_sz, 0, 0);

	int i;
	for (i = 0; i < best_sz; i++) {
		best_nums[i] = best[i].node_num;
	}
}
