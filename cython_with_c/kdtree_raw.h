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

#define LIMIT 3

/* Structs */
typedef struct kdtree_node kdtree_node;

struct kdtree_node{
	double * coords;
  int number;
	kdtree_node *left;
  kdtree_node *right;
};

typedef struct point_data {
	int number;
	double *coords;
} point_data;

/* prototypes */
extern void run_nn_search(kdtree_node *root, 
		int best_nums[], 
		int best_sz, 
		int search_num, 
		double search[]);
