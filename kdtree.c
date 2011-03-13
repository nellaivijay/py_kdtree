/*
 * Copyright 2011 Chris M Bouzek
 */
#include <Python.h>
#include "structmember.h"

#define LIMIT 3

/* Structs */

typedef struct {
    PyObject_HEAD
    PyObject *left;
    PyObject *right;
		double xcoord;
		double ycoord;
    int number;
} KDTreeNode;

struct best_pair {
	int node_num;
	double dist;
};

/* prototypes */

static PyObject *
KDTreeNode_getright(KDTreeNode *self, void *closure);

static int
KDTreeNode_setright(KDTreeNode *self, PyObject *value, void *closure);

static PyObject *
KDTreeNode_getleft(KDTreeNode *self, void *closure);

static int
KDTreeNode_setleft(KDTreeNode *self, PyObject *value, void *closure);

static PyObject *
KDTreeNode_run_nn_search(KDTreeNode *self, PyObject *args);

/* Utility functions */

/* Determine the largest element in the best neighbors array*/
static double largest_dist(struct best_pair best[], int count) {
  if (count < 1) {
    return -1;
	}
  if (count < LIMIT) {
    return best[count - 1].dist;
	}
  /* 2 = 3 - 1 */
  return best[LIMIT - 1].dist;
}

static double sqdist(double ax, double bx, double ay, double by) {
	double diffx = ax - bx;
	double diffy = ay - by;
	return (diffx * diffx) + (diffy * diffy);
}

/* Functions directly related to KD-tree functionality */

/* Adds the point to the list of best nodes if it is closer than the current best
   list. */
static int add_best(
		int count, 
		double nodepoint[], 
		int node_num, 
		double point[], 
		struct best_pair best[]) {
	/*(count, nodepoint, node_num, point, point_num, best)*/
  /* due to the constraints of the problem, we need to check each node
     before assigning it as the final best choice to ensure it is not
     equal to the searched-for point
	*/

	double sd = sqdist(nodepoint[0], nodepoint[1], point[0], point[1]);
	int last_idx;
	if (count < LIMIT) {
		last_idx = count;
	} else {
		last_idx = count - 1;
	}

	int idx;
	struct best_pair candidate;
	candidate.node_num = node_num;
	candidate.dist = sd;

	struct best_pair pair;
	/* search through linearly to maintain sorted order */
	for (idx = 0; idx < count; idx++) {
		pair = best[idx];
		if (pair.dist > sd) {
			/* push elements down */
			int x;
			/*free(best[last_idx]);*/
			for (x = last_idx; x > idx; x--) {
				best[x] = best[x - 1];
			}
			/* TODO does this need to be allocated on the heap? */
			best[idx] = candidate;
			return last_idx + 1;
		}
	}

	if (count < LIMIT) {
		/* didn't find an insert spot, so insert at the end */
		best[count] = candidate;
		return last_idx + 1;
	}
  return count;
}

/* Searches for nearest neighbor of point using node as the root. */
static int nn_search(
		KDTreeNode *node, 
		int point_num,
		double search_x,
		double search_y,
		struct best_pair best[], 
		int count, 
		int axis) {
  if (Py_None == (PyObject *)node) {
    return count;
	}

	int cnt = count;
  double nodepoint[2];
	nodepoint[0] = node->xcoord;
	nodepoint[1] = node->ycoord;
	double point[2];
	point[0] = search_x;
	point[1] = search_y;
	int node_num = node->number;

	if (Py_None == node->left && Py_None == node->right) {
    if (node_num != point_num) {
      cnt = add_best(cnt, nodepoint, node_num, point, best);
		}
    return cnt;
	}

  /* Normally we'd select axis based on depth so that axis cycles through 
     all valid values.  Here we know we have 2-dimensional values, so don't 
     waste time calculating, but just alternate
     axis = depth % len(point) */

  /* compare query point and current node along the axis to see which tree is
     far and which is near */

	KDTreeNode *near;
	KDTreeNode *far;
  if (point[axis] < nodepoint[axis]) {
    near = (KDTreeNode *)node->left;
    far = (KDTreeNode *)node->right;
	} else {
    near = (KDTreeNode *)node->right;
    far = (KDTreeNode *)node->left;
	}

  /* search the near branch */
  int next_axis;
	if (1 == axis) {
		next_axis = 0;
	} else {
		next_axis = 1;
	}
  cnt = nn_search(near, point_num, search_x, search_y, best, cnt, next_axis);

  /* If the current node is closer overall than the current best */
  if (node_num != point_num) {
    cnt = add_best(cnt, nodepoint, node_num, point, best);
	}

  /* maybe search the away branch */
  int largest = largest_dist(best, cnt);
	double point_dist = nodepoint[axis] - point[axis];
  if ((-1 == largest) || (point_dist * point_dist < largest)) {
    cnt = nn_search(far, point_num, search_x, search_y, best, cnt, next_axis);
	}
  return cnt;
}

/* Ref counting cycle detection */
static int
KDTreeNode_traverse(KDTreeNode *self, visitproc visit, void *arg) {
	int vret;

	if (self->left) {
		vret = visit(self->left, arg);
		if (vret != 0) {
			return vret;
		}
	}
	if (self->right) {
		vret = visit(self->right, arg);
		if (vret != 0) {
			return vret;
		}
	}

	return 0;
}

static int 
KDTreeNode_clear(KDTreeNode *self) {
	PyObject *tmp;

	tmp = self->left;
	self->left = NULL;
	Py_XDECREF(tmp);

	tmp = self->right;
	self->right = NULL;
	Py_XDECREF(tmp);

	return 0;
}

static void
KDTreeNode_dealloc(KDTreeNode* self) {
	KDTreeNode_clear(self);
	self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
KDTreeNode_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
	KDTreeNode *self;

	self = (KDTreeNode *)type->tp_alloc(type, 0);
	if (self != NULL) {
		/* TODO do we need a decref on the Py_None objects? */
		self->left = Py_None;
		self->right = Py_None;
		self->xcoord = 0.0;
		self->ycoord = 0.0;
		self->number = 0;
	}

	return (PyObject *)self;
}

static int
KDTreeNode_init(KDTreeNode *self, PyObject *args, PyObject *kwds) {
	PyObject *left=NULL, *right=NULL, *tmp;

	static char *kwlist[] = {"number", "coords", "left", "right", NULL};

	if (! PyArg_ParseTupleAndKeywords(args, kwds, "i(dd)|OO", kwlist, 
																		&self->number, 
																		&self->xcoord, &self->ycoord, 
																		&left, &right)) {
		return -1; 
	}

	/* TODO use
	 * if (! PyObject_TypeCheck(some_object, &MyType)) {
	 *     PyErr_SetString(PyExc_TypeError, "arg #1 not a mything");
	 *         return NULL;
	 *         }
	 *
	 * on the left and right */

	if (left) {
		tmp = self->left;
		Py_INCREF(left);
		self->left = left;
		Py_XDECREF(tmp);
	}

	if (right) {
		tmp = self->right;
		Py_INCREF(right);
		self->right = right;
		Py_XDECREF(tmp);
	}


	return 0;
}

static PyMemberDef KDTreeNode_members[] = {
	{"xcoord", T_DOUBLE, offsetof(KDTreeNode, xcoord), 0,
	 "X coordinate"},
	{"ycoord", T_DOUBLE, offsetof(KDTreeNode, ycoord), 0,
	 "Y coordinate"},
	{"number", T_INT, offsetof(KDTreeNode, number), 0,
	 "kd-tree node number"},
	{NULL}  /* Sentinel */
};

static PyGetSetDef KDTreeNode_getseters[] = {
	{"left", 
	 (getter)KDTreeNode_getleft, (setter)KDTreeNode_setleft,
	 "left node",
	 NULL},
	{"right", 
	 (getter)KDTreeNode_getright, (setter)KDTreeNode_setright,
	 "right node",
	 NULL},
	{NULL}  /* Sentinel */
};


static PyObject *
KDTreeNode_point(KDTreeNode* self) {
	static PyObject *format = NULL;
	PyObject *args, *result;

	if (format == NULL) {
		format = PyString_FromString("%d %d");
		if (format == NULL) {
			return NULL;
		}
	}

	args = Py_BuildValue("dd", self->xcoord, self->ycoord);

	if (args == NULL) {
		return NULL;
	}

	result = PyString_Format(format, args);
	Py_DECREF(args);
	
	return result;
}

static PyMethodDef KDTreeNode_methods[] = {
	{"point", (PyCFunction)KDTreeNode_point, METH_NOARGS,
	 "Return the points as a string"
	},
	{"run_nn_search", (PyCFunction)KDTreeNode_run_nn_search, METH_VARARGS,
	 "Runs a nearest-neighbor search"
	},
	{NULL}  /* Sentinel */
};

static PyTypeObject KDTreeNodeType = {
	PyObject_HEAD_INIT(NULL)
	0,                         /*ob_size*/
	"kdtree.KDTreeNode",            /*tp_name*/
	sizeof(KDTreeNode),             /*tp_basicsize*/
	0,                         /*tp_itemsize*/
	(destructor)KDTreeNode_dealloc, /*tp_dealloc*/
	0,                         /*tp_print*/
	0,                         /*tp_getattr*/
	0,                         /*tp_setattr*/
	0,                         /*tp_compare*/
	0,                         /*tp_repr*/
	0,                         /*tp_as_number*/
	0,                         /*tp_as_sequence*/
	0,                         /*tp_as_mapping*/
	0,                         /*tp_hash */
	0,                         /*tp_call*/
	0,                         /*tp_str*/
	0,                         /*tp_getattro*/
	0,                         /*tp_setattro*/
	0,                         /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /*tp_flags*/
	"KDTreeNode objects",           /* tp_doc */
	(traverseproc)KDTreeNode_traverse,   /* tp_traverse */
	(inquiry)KDTreeNode_clear,           /* tp_clear */
	0,		               /* tp_richcompare */
	0,		               /* tp_weaklistoffset */
	0,		               /* tp_iter */
	0,		               /* tp_iternext */
	KDTreeNode_methods,             /* tp_methods */
	KDTreeNode_members,             /* tp_members */
	KDTreeNode_getseters,     /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)KDTreeNode_init,      /* tp_init */
	0,                         /* tp_alloc */
	KDTreeNode_new,                 /* tp_new */
};

/* Initializes the nearest neighbor search point and starts the search.
 * call it like run_nn_search(kdtree_node, search_num, (search_x, search_y))
 */
static PyObject *
KDTreeNode_run_nn_search(KDTreeNode *self, PyObject *args) {
	/* The number of the node we are running the nearest neighbor search on */
	int search_num; 
	/* The X,Y coordinates of the node we are running nearest neighbor search on */
	double search_x;
	double search_y;
	if(!PyArg_ParseTuple(args, "i(dd)", &search_num, &search_x, &search_y)) {
		return NULL;
	}

	struct best_pair best[LIMIT];
	int count = nn_search(
			self, search_num, search_x, search_y, best, 0, 0);
	return PyInt_FromLong(count);
}

static PyObject *
KDTreeNode_getleft(KDTreeNode *self, void *closure)
{
	Py_INCREF(self->left);
	return self->left;
}

static int
KDTreeNode_setleft(KDTreeNode *self, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the left attribute");
    return -1;
  }
  
	if (value != Py_None && !PyObject_TypeCheck(value, &KDTreeNodeType)) {
    PyErr_SetString(PyExc_TypeError, 
                    "The left attribute value must be a KDTreeNode");
    return -1;
  }
      
  Py_DECREF(self->left);
  Py_INCREF(value);
  self->left = value;    

  return 0;
}

static PyObject *
KDTreeNode_getright(KDTreeNode *self, void *closure)
{
	Py_INCREF(self->right);
	return self->right;
}

static int
KDTreeNode_setright(KDTreeNode *self, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the right attribute");
    return -1;
  }
  
	if (value != Py_None && !PyObject_TypeCheck(value, &KDTreeNodeType)) {
    PyErr_SetString(PyExc_TypeError, 
                    "The right attribute value must be a KDTreeNode");
    return -1;
  }
      
  Py_DECREF(self->right);
  Py_INCREF(value);
  self->right = value;    

  return 0;
}

static PyMethodDef module_methods[] = {
	{NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initkdtree(void) {
	PyObject* m;

	if (PyType_Ready(&KDTreeNodeType) < 0) {
		return;
	}

	m = Py_InitModule3("kdtree", module_methods,
										 "A simple 2-dimensional KDTreeNode extension to Python.");

	if (m == NULL) {
		return;
	}

	Py_INCREF(&KDTreeNodeType);
	PyModule_AddObject(m, "KDTreeNode", (PyObject *)&KDTreeNodeType);
}

