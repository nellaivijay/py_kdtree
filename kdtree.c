#include <Python.h>
#include "structmember.h"

typedef struct {
    PyObject_HEAD
    PyObject *left;
    PyObject *right;
		double xcoord;
		double ycoord;
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
	{"left", T_OBJECT_EX, offsetof(KDTreeNode, left), 0,
	 "left node"},
	{"right", T_OBJECT_EX, offsetof(KDTreeNode, right), 0,
	 "right node"},
	{"xcoord", T_DOUBLE, offsetof(KDTreeNode, xcoord), 0,
	 "X coordinate"},
	{"ycoord", T_DOUBLE, offsetof(KDTreeNode, ycoord), 0,
	 "Y coordinate"},
	{"number", T_INT, offsetof(KDTreeNode, number), 0,
	 "kd-tree node number"},
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
	0,                         /* tp_getset */
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

