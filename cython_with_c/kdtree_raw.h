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

/* Structs */

/**
 * A container for k-dimensional space point data.  This differs from kdtree_node 
 * in that it is only used for input and searching for nearest neighbor.
 * @param num The node number for the input data.  Should match with a kdtree_node
 * num.
 * @param coords The coordinates in the k-dimensional space.
 * @param dims The number of dimensions.
 * @param curr_axis The current axis being searched/split upon.
 */
typedef struct point_data {
	int num;
	double *coords;
	size_t dims;
	size_t curr_axis;
} point_data;

typedef struct kdtree_node kdtree_node;
/**
 * A node representing a KD tree.
 * @param point_data The information regarding node's point in k-dimensional space.
 * @param left The node's left child.
 * @param right The node's right child.
 */
struct kdtree_node{
	point_data *data;
	kdtree_node *left;
  kdtree_node *right;
};

/* prototypes */
extern void run_nn_search(kdtree_node *root, 
		size_t num_neighbors, 
		point_data pd, 
		int best_nums[]);

extern kdtree_node * fill_tree(point_data **points, size_t num_points);

extern void free_tree(kdtree_node * node);

extern double sqdist(double a[], double b[], size_t dims);
