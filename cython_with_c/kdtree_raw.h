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
typedef struct kdtree_node kdtree_node;

struct kdtree_node{
	double *coords;
  int num;
	kdtree_node *left;
  kdtree_node *right;
};

typedef struct point_data {
	int num;
	int sz;
	int curr_axis;
	double *coords;
} point_data;

/* prototypes */
extern void run_nn_search(kdtree_node *root, 
		int num_neighbors, 
		point_data pd, 
		int best_nums[]);

extern kdtree_node * fill_tree(point_data **points, int num_points, int dims);

extern void free_tree(kdtree_node * node);
