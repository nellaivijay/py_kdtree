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

/**
 * Represents a neighbor of an arbitrary node.  This is a combination of node 
 * number and distance to said arbitrary node.
 */
typedef struct best_pair {
	int node_num;
	double dist;
} best_pair;

/* Utility functions */

/**
 * Determine the largest element in the nearest neighbors array.
 * @param [in] nearest The array of nearest neighbors.
 * @param [in] count The number of current nearest neighbors.
 * @return The largest value in the nearest neighbors array (i.e. the
 * neighbor that is farthest away).
 */
static double largest_dist(best_pair nearest[], size_t count) {
	double largest = -1.0;
	if (count > 0) {
		largest = nearest[count - 1].dist;
	}
	return largest;
}

/**
 * Choose the axis to use based on the current depth and the number of dimensions.
 * This is useful when building the tree and search for nearest neighbors.
 * @param [in] depth The current depth in the tree.
 * @param [in] dims The number of dimensions in the points used for the tree.
 * @return The axis to use.
 */
static size_t pick_axis(size_t depth, size_t dims) {
	return depth % dims;
}

/**
 * Computes the Euclidean distance between two k-dimensional points.
 * @param [in] a The first point.
 * @param [in] b The second point.
 * @param [in] dims The number of dimensions in each point.
 * @return The Euclidean distance between a and b.
 */
extern double sqdist(double a[], double b[], size_t dims) {
	size_t k;
	double dist = 0.0;
	double diff;
	for (k = 0; k < dims; k++) {
		diff = a[k] - b[k];
		dist += (diff * diff);
	}
	return dist;
}

/**
 * Compares two point_data items based on the coordinate and their current axis.
 * @param [in] a The first point_data item to compare.  The curr_axis member must be 
 * set by the caller.
 * @param [in] b The second point_data item to compare.  The curr_axis member must be 
 * set by the caller.
 * @return -1 if a.coords[curr_axis] < b.coords[curr_axis], 
 * 1 if a.coords[curr_axis] > b.coords[curr_axis], 0 otherwise.
 */
static int comp_axis(const void *a, const void *b) {
	point_data * const *pa = a;
	point_data * const *pb = b;

	size_t axis = (*pa)->curr_axis;
	double a_coord = (*pa)->coords[axis];
	double b_coord = (*pb)->coords[axis];
	if (a_coord < b_coord) {
		return -1;
	}
	if (a_coord > b_coord) {
		return 1;
	}
	return 0;
}

/**
 * Debug method to print out the points array.
 * @param [in] points The points array to print.
 * @param [in] sz The number of points in the array.
 */
static void print_points(const point_data **points, size_t sz) {
	size_t y;
	for (y = 0; y < sz; y++) {
		fprintf(stderr, "cpoint %f, %f\n", points[y]->coords[0], points[y]->coords[1]);
	}
}

/**
 * Slices the points_data array.
 * Note that this does not perform a deep copy, but just copies the pointers so
 * they can be reordered without touching the original array.
 * @param [in] points The array of points to slice.
 * @param [in] start The starting point of the slice (inclusive).
 * @param [in] end The ending point of the slice (exclusive).
 * @return The slice of the array.
 */
static point_data **slice(point_data **points, size_t start, size_t end) {
	size_t slice_sz = end - start;
	size_t cp_sz = slice_sz * sizeof(point_data *);
	point_data **sliced = malloc(cp_sz);
	if (NULL == sliced) {
		fprintf(stderr, "Out of memory at %s: %d\n", __FILE__, __LINE__);
		exit(OOM);
	}
	memcpy(sliced, &(points[start]), cp_sz);
	return sliced;
}

/**
 * Builds up a tree using the given point_data.  
 * @param [in] points The points_data used to build the tree.  This function copies 
 * the pointers in this parameter so the caller is free to dispose of it after 
 * the call.
 * @param [in] num_points The number of points in the points_data array.
 * @param [in] depth The current depth of the tree.  Used to correctly sort and 
 * split the points
 * @return A newly malloc'd KD tree node.
 */
static kdtree_node * fill_tree_r(point_data **points, size_t num_points, size_t depth) {
	if (NULL == points || 0 == num_points) {
		return NULL;
	}

	size_t dims = points[0]->sz;

	kdtree_node *node = malloc(sizeof(kdtree_node));
	if (NULL == node) {
		fprintf(stderr, "Out of memory at %s: %d\n", __FILE__, __LINE__);
		exit(OOM);
	}
	node->left = NULL;
	node->right = NULL;

	size_t axis = pick_axis(depth, dims);
	/* Sort the points by axis; we need to mod the current axis before sorting to get
	 * the comparison to work correctly */
	size_t x;
	for (x = 0; x < num_points; x++) {
		points[x]->curr_axis = axis;
	}
	qsort(points, num_points, sizeof(*points), comp_axis);

	/* Print input */

	size_t median = num_points / 2;
	size_t left_sz = median;
	size_t right_sz = num_points - median - 1;

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

	/* Now divide and recurse left/right */
	size_t next_depth = depth + 1;
	if (left_sz > 0) {
		/* Left side goes from [0, median), i.e. does not include the median */
		point_data **left_arr = slice(points, 0, median);
		node->left = fill_tree_r(left_arr, median, next_depth);
		free(left_arr);
	}

	/* Right side goes from [median + 1, num_points).  The current node is the median, and
	 * we run up to the last element in the subarray.*/
	if (right_sz > 0) {
		point_data **right_arr = slice(points, median + 1, num_points);
		node->right = fill_tree_r(right_arr, right_sz, next_depth);
		free(right_arr);
	}
	return node;
}

/**
 * Builds up a tree using the given point_data.  
 * @param [in] points The points_data used to build the tree.  This function copies 
 * the pointers in this parameter so the caller is free to dispose of it after 
 * the call.
 * @param [in] num_points The number of points in the points_data array.
 * @return A newly malloc'd KD tree node.
 */
extern kdtree_node * fill_tree(point_data **points, size_t num_points) {
	return fill_tree_r(points, num_points, 0);
}

/**
 * Recursively free()s the elements of the tree.
 * @param [in] node The root of the tree to free.
 */
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

/**
 * Adds the search point to the list of nearest neighbors if it is closer than 
 * any of the current nearest neighbors. 
 * @param [in] nearest The current nearest neighbors.  Will be filled in
 * by this function.
 * @param [in] best_count The number of current nearest neighbors.
 * @param [in] neighbor The node to consider as a potential nearest neighbor.
 * @param [in] search The point for which the nearest neighbor search is being
 * done.
 * @param [in] num_neighbors The maximum number of nearest neighbors.
 * @return The number of current nearest neighbors.  If best_count < num_neigbors,
 * this will be one more than best_count; otherwise it will be equal to 
 * num_neighbors.
 */
static size_t add_best(
		best_pair nearest[],
		size_t best_count, 
		const kdtree_node *neighbor,
		point_data search, 
		size_t num_neighbors) {

	double sd = sqdist(neighbor->coords, search.coords, search.sz);
	size_t last_idx;
	if (best_count < num_neighbors) {
		last_idx = best_count;
	} else {
		last_idx = best_count - 1;
	}

	size_t idx;
	best_pair candidate;
	candidate.node_num = neighbor->num;
	candidate.dist = sd;

	best_pair pair;
	size_t x;
	/* search through linearly to maintain sorted order */
	for (idx = 0; idx < best_count; idx++) {
		pair = nearest[idx];
		if (pair.dist > sd) {
			/* push elements down */
			for (x = last_idx; x > idx; x--) {
				nearest[x] = nearest[x - 1];
			}
			nearest[idx] = candidate;
			return last_idx + 1;
		}
	}

	if (best_count < num_neighbors) {
		/* didn't find an insert spot, so insert at the end */
		nearest[last_idx] = candidate;
		return last_idx + 1;
	}
  return best_count;
}

/**
 * Searches for nearest neighbor of search using node as the root.
 * @param [in] node The node to consider as a potential nearest neighbor.
 * @param [in] search The point for which the nearest neighbor search is being
 * done.
 * @param [in] nearest The current nearest neighbors.  Will be filled in
 * by this function.
 * @param [in] best_count The number of current nearest neighbors.
 * @param [in] num_neighbors The maximum number of nearest neighbors.
 * @param [depth] The current depth in the search tree.
 *
 * @return The number of current nearest neighbors.  If best_count < num_neigbors,
 * this will be one more than best_count; otherwise it will be equal to 
 * num_neighbors.
 */
static size_t nn_search(
		const kdtree_node *node, 
		point_data search,
		best_pair nearest[], 
		size_t best_count, 
		size_t num_neighbors,
		size_t depth) {
  if (NULL == node) {
    return best_count;
	}
	
	int search_num = search.num;
	size_t dims = search.sz;
	size_t axis = pick_axis(depth, dims);

	int node_num = node->num;
	double neighbor_coord = node->coords[axis];
	double search_coord = search.coords[axis];

  /* we need to check each node before assigning it as the final best choice to 
	   ensure it is not equal to the searched-for point, hence 
	   node_num != search_num */
	if (NULL == node->left && NULL == node->right) {
    if (node_num != search_num) {
      best_count = add_best(nearest, best_count, node, search, num_neighbors);
		}
    return best_count;
	}

  /* compare query point and current node along the axis to see which tree is
     far and which is near */

	kdtree_node *near;
	kdtree_node *far;
  if (search_coord < neighbor_coord) {
    near = (kdtree_node *)node->left;
    far = (kdtree_node *)node->right;
	} else {
    near = (kdtree_node *)node->right;
    far = (kdtree_node *)node->left;
	}

  /* search the near branch */
  size_t next_depth = depth + 1;

	if (NULL != near) {
	  best_count = nn_search(near, search, nearest, best_count, num_neighbors, next_depth);
	}

  /* If the current node is closer overall than the current best */
  if (node_num != search_num) {
    best_count = add_best(nearest, best_count, node, search, num_neighbors);
	}

  /* maybe search the away branch */
	if (NULL != far) {
		double largest = largest_dist(nearest, best_count);
		size_t search_other = 0;
		if (largest < 0) {
			search_other = 1;
		} else {
			double diff = neighbor_coord - search_coord;
			if ((diff * diff) < largest) {
				search_other = 1;
			}
		}
		if (1 == search_other) {
			best_count = nn_search(far, search, nearest, best_count, num_neighbors, next_depth);
		}
	}
  return best_count;
}

/** 
 * Initializes the nearest neighbor search point and starts the search.
 *
 * @param [in] root The node to start the nearest neighbor search at.
 * @param [in] search The point for which the nearest neighbor search is being
 * done.
 * @param [in] best_nums The nearest neighbors node numbers.  Will be filled in
 * by this function.
 */
extern void
run_nn_search(kdtree_node *root, 
		size_t num_neighbors, 
		point_data search,
		int best_nums[]) {
	best_pair nearest[num_neighbors];
	nn_search(root, search, nearest, 0, num_neighbors, 0);

	size_t i;
	for (i = 0; i < num_neighbors; i++) {
		best_nums[i] = nearest[i].node_num;
	}
}
