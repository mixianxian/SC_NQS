from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "rebuild",
        ["rebuild.pyx"],
        language='c',
        include_dirs=[numpy.get_include()],
    )
]
setup(
    name='rebuild',
    ext_modules=cythonize(ext_modules,
    compiler_directives={'language_level':3,'boundscheck':False,'wraparound':True},
    annotate=True))
