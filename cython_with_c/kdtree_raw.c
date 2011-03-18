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
#include <stdlib.h>
#include <string.h>
#include "kdtree_raw.h"

#ifndef OOM
#define OOM 8
#endif

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

/*
 * Compares two point_data items based on the coordinate and their current axis.
 */
static int comp_axis(const void *a, const void *b) {
	point_data * const *pa = a;
	point_data * const *pb = b;

	int axis = (*pa)->curr_axis;
	double a_coord = (*pa)->coords[axis];
	double b_coord = (*pb)->coords[axis];
	/*fprintf(stderr, "a coord %f\n", a_coord);
	fprintf(stderr, "b coord %f\n", b_coord);*/
	if (a_coord < b_coord) {
		return -1;
	}
	if (a_coord > b_coord) {
		return 1;
	}
	return 0;
}

static void print_points(point_data **points, int num_points) {
	size_t y;
	for (y = 0; y < num_points; y++) {
		fprintf(stderr, "cpoint %f, %f\n", points[y]->coords[0], points[y]->coords[1]);
	}
}

/*
 * Builds up a tree using the given point_data.  This copies the data in points[] so the 
 * caller is free to dispose of it after the call.
 */
extern kdtree_node * fill_tree(point_data **points, int num_points, int dims, int axis) {
	if (NULL == points || 0 == num_points) {
		return NULL;
	}

	kdtree_node *node = malloc(sizeof(kdtree_node));
	if (NULL == node) {
		fprintf(stderr, "Out of memory at %s: %d\n", __FILE__, __LINE__);
		exit(OOM);
	}
	node->left = NULL;
	node->right = NULL;

	/* Sort the points by axis; we need to mod the current axis before sorting to get
	 * the comparison to work correctly */
	size_t x;
	for (x = 0; x < num_points; x++) {
		points[x]->curr_axis = axis;
	}
	qsort(points, num_points, sizeof(*points), comp_axis);

	/* Print input */
	/*fprintf(stderr, "========== Num points after sort: %d\n", num_points);
	print_points(points, num_points);*/

	int next_axis = pick_axis(axis);
	int median = num_points / 2;
	int left_sz = median;
	/*fprintf(stderr, "Median %d\n", median);*/
	int right_sz = num_points - median - 1;

	point_data *p_median = points[median];
	node->num = p_median->num;

	size_t coord_size = dims * sizeof(double);
	node->coords = malloc(coord_size);
	if (NULL == node->coords) {
		fprintf(stderr, "Out of memory at %s: %d\n", __FILE__, __LINE__);
		exit(OOM);
	}
	/* copy points over */
	memcpy(node->coords, p_median->coords, coord_size);
	/*fprintf(stderr, "Current node: index %d, number %d, coords (%f, %f)\n", median, node->num, node->coords[0], node->coords[1]);*/

	/*if (left_sz > 0) {
		fprintf(stderr, "======== left copy: [%d, %d)\n", 0, left_sz);
	}
	if (right_sz > 0) {
		fprintf(stderr, "======== right copy: [%d, %d)\n", median + 1, median + 1 + right_sz);
	}*/
	/* Now divide and recurse left/right */
	if (left_sz > 0) {
		/* Left side goes from [0, median), i.e. does not include the median */
		size_t cp_sz = left_sz * sizeof(point_data *);
		point_data **left_arr = malloc(cp_sz);
		if (NULL == left_arr) {
			fprintf(stderr, "Out of memory at %s: %d\n", __FILE__, __LINE__);
			exit(OOM);
		}
		/*fprintf(stderr, "starting left copy for %d points\n", left_sz);*/
		memcpy(left_arr, points, cp_sz);
		/*print_points(left_arr, left_sz);*/
		/*fprintf(stderr, "Recursing for left\n");*/
		node->left = fill_tree(left_arr, median, dims, next_axis);
		free(left_arr);
		/*fprintf(stderr, "finished left copy\n");*/
	}
	/* Right side goes from [median + 1, num_points).  The current node is the median, and
	 * we run up to the last element in the subarray.*/

	if (right_sz > 0) {
		size_t cp_sz = right_sz * sizeof(point_data *);
		point_data **right_arr = malloc(cp_sz);
		if (NULL == right_arr) {
			fprintf(stderr, "Out of memory at %s: %d\n", __FILE__, __LINE__);
			exit(OOM);
		}
		/*fprintf(stderr, "Recursing for right\n");*/
		memcpy(right_arr, &(points[median + 1]), cp_sz);
		node->right = fill_tree(right_arr, right_sz, dims, next_axis);
		free(right_arr);
		/*fprintf(stderr, "finished right copy\n");*/
	}
	return node;
}

extern void free_tree(kdtree_node * node) {
	if (NULL == node) {
		return;
	}
	if (NULL != node->right) {
		free_tree(node->right);
	}
	if (NULL != node->left) {
		free_tree(node->left);
	}
	if (NULL != node->coords) {
		free(node->coords);
	}
	free(node);
	/* TODO think this needs to be a pointer to a pointer for this NULL to work. */
	node = NULL;
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
		int num_neighbors) {

	double sd = sqdist(nodepoint, search);
	int last_idx;
	if (count < num_neighbors) {
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

	if (count < num_neighbors) {
		/* didn't find an insert spot, so insert at the end */
		best[last_idx] = candidate;
		return last_idx + 1;
	}
  return count;
}

/* Searches for nearest neighbor of search using node as the root. */
static int nn_search(
		const kdtree_node *node, 
		int search_num,
		double search[],
		best_pair best[], 
		int num_neighbors,
		int count, 
		int axis) {
  if (NULL == node) {
    return count;
	}

	int cnt = count;
  double *nodepoint = node->coords;
	int node_num = node->num;

  /* due to the constraints of the problem, we need to check each node
     before assigning it as the final best choice to ensure it is not
     equal to the searched-for point, hence node_num != search_num */
	if (NULL == node->left && NULL == node->right) {
    if (node_num != search_num) {
      cnt = add_best(cnt, nodepoint, node_num, search, best, num_neighbors);
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
	  cnt = nn_search(near, search_num, search, best, num_neighbors, cnt, next_axis);
	}

  /* If the current node is closer overall than the current best */
  if (node_num != search_num) {
    cnt = add_best(cnt, nodepoint, node_num, search, best, num_neighbors);
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
			cnt = nn_search(far, search_num, search, best, num_neighbors, cnt, next_axis);
		}
	}
  return cnt;
}

/* Initializes the nearest neighbor search point and starts the search.
 * call it like run_nn_search(kdtree_node, search_num, (search_x, search_y))
 */
extern void
run_nn_search(kdtree_node *root, 
		int best_nums[], 
		int num_neighbors, 
		int search_num, 
		double search[]) {
	best_pair best[num_neighbors];
	nn_search(root, search_num, search, best, num_neighbors, 0, 0);

	int i;
	for (i = 0; i < num_neighbors; i++) {
		best_nums[i] = best[i].node_num;
	}
}
