from distutils.core import setup, Extension
from Cython.Distutils import build_ext
#from distutils.extension import Extension

sourcefiles = ['kdtree.pyx', 'kdtree_raw.c']
#sourcefiles = ['kdtree_raw.c']

setup(
  name="kdtree", version="1.0",
  cmdclass = {'build_ext': build_ext},
  ext_modules = [Extension("kdtree", sourcefiles)]
)

