#include <Python.h>
#include "structmember.h"

typedef struct {
    PyObject_HEAD
    PyObject *left;
    PyObject *right;
		PyTupleObject *coords;
    int number;
} KDTreeNode;

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
	PyTupleObject *ctmp;

	tmp = self->left;
	self->left = NULL;
	Py_XDECREF(tmp);

	tmp = self->right;
	self->right = NULL;
	Py_XDECREF(tmp);

	ctmp = self->coords;
	self->coords = NULL;
	Py_XDECREF(ctmp);

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
		self->coords = (PyTupleObject *) Py_None;
		self->number = 0;
	}

	return (PyObject *)self;
}

static int
KDTreeNode_init(KDTreeNode *self, PyObject *args, PyObject *kwds) {
	PyObject *left=NULL, *right=NULL, *tmp;
	PyTupleObject *coords=NULL, *ctmp;

	static char *kwlist[] = {"number", "coords", "left", "right", NULL};

	if (! PyArg_ParseTupleAndKeywords(args, kwds, "iO|OO", kwlist, 
																		&self->number, &coords, &left, &right)) {
		return -1; 
	}

	/* TODO use
	 * if (! PyObject_TypeCheck(some_object, &MyType)) {
	 *     PyErr_SetString(PyExc_TypeError, "arg #1 not a mything");
	 *         return NULL;
	 *         }
	 *
	 * on the left and right */
	if (coords) {
		if (!PyTuple_Check(coords)) {
			PyErr_SetString(PyExc_TypeError, "coords must be a tuple");
			return -1;
		}
		PyObject *xcoord;
		xcoord = PyTuple_GetItem((PyObject *)coords, 0);

		if (!PyFloat_Check(xcoord)) {
			PyErr_SetString(PyExc_TypeError, "X coordinate must be a float");
			Py_XDECREF(xcoord);
			return -1;
		}
		Py_XDECREF(xcoord);

		PyObject *ycoord;
		ycoord = PyTuple_GetItem((PyObject *)coords, 1);

		if (!PyFloat_Check(ycoord)) {
			PyErr_SetString(PyExc_TypeError, "Y coordinate must be a float");
			Py_XDECREF(ycoord);
			return -1;
		}
		Py_XDECREF(ycoord);

		ctmp = self->coords;
		Py_INCREF(coords);
		self->coords = coords;
		Py_XDECREF(ctmp);
	}

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
	{"left", T_OBJECT_EX, offsetof(KDTreeNode, left), 0,
	 "left node"},
	{"right", T_OBJECT_EX, offsetof(KDTreeNode, right), 0,
	 "right node"},
	{"number", T_INT, offsetof(KDTreeNode, number), 0,
	 "kd-tree node number"},
	{NULL}  /* Sentinel */
};

static PyObject *
KDTreeNode_point(KDTreeNode* self) {
	static PyObject *format = NULL;
	PyObject *args, *result;

	if (format == NULL) {
		format = PyString_FromString("%s %s");
		if (format == NULL) {
			return NULL;
		}
	}

	if (self->coords == NULL) {
		PyErr_SetString(PyExc_AttributeError, "coords");
		return NULL;
	}

	PyObject *xcoord;
	xcoord = PyTuple_GetItem((PyObject *)self->coords, 0);

	PyObject *ycoord;
	ycoord = PyTuple_GetItem((PyObject *)self->coords, 1);

	args = Py_BuildValue("OO", xcoord, ycoord);

	Py_XDECREF(xcoord);
	Py_XDECREF(ycoord);
	if (args == NULL) {
		return NULL;
	}

	result = PyString_Format(format, args);
	Py_DECREF(args);
	
	return result;
}

static PyTupleObject *
KDTreeNode_getcoords(KDTreeNode *self, void *closure) {
	Py_INCREF(self->coords);
	return self->coords;
}

static int
KDTreeNode_setcoords(KDTreeNode *self, PyTupleObject *coords, void *closure) {
  if (coords == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot set coordinates to null.");
    return -1;
  }

	if (!PyTuple_Check(coords)) {
		PyErr_SetString(PyExc_TypeError, "coords must be a tuple");
		return -1;
	}
	PyObject *xcoord;
	xcoord = PyTuple_GetItem((PyObject *)coords, 0);

	if (!PyFloat_Check(xcoord)) {
		PyErr_SetString(PyExc_TypeError, "X coordinate must be a float");
		Py_XDECREF(xcoord);
		return -1;
	}
	Py_XDECREF(xcoord);

	PyObject *ycoord;
	ycoord = PyTuple_GetItem((PyObject *)coords, 1);

	if (!PyFloat_Check(ycoord)) {
		PyErr_SetString(PyExc_TypeError, "Y coordinate must be a float");
		Py_XDECREF(ycoord);
		return -1;
	}
	Py_XDECREF(ycoord);

	PyTupleObject *ctmp;
	ctmp = self->coords;
	Py_INCREF(coords);
	self->coords = coords;
	Py_XDECREF(ctmp);
  
  return 0;
}

static PyGetSetDef KDTreeNode_getseters[] = {
	{"coords", 
	 (getter)KDTreeNode_getcoords, (setter)KDTreeNode_setcoords,
	 "coordinates",
	 NULL},
	{NULL}  /* Sentinel */
};


static PyMethodDef KDTreeNode_methods[] = {
	{"point", (PyCFunction)KDTreeNode_point, METH_NOARGS,
	 "Return the points as a string"
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
	KDTreeNode_getseters,      /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)KDTreeNode_init,      /* tp_init */
	0,                         /* tp_alloc */
	KDTreeNode_new,                 /* tp_new */
};

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

