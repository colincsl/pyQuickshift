

# python setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np      


setup(cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("pyQuickShift", ["pyQuickShift.pyx", "../quickshift.c", "../generic.c",\
     		"../mathop.c", "../mathop_sse2.c", "../imopv.c", "../imopv_sse2.c", "../host.c",\
     		"../array.c", "../getopt_long.c", "../random.c"],
        language="c", libraries=["m"])],
    include_dirs = [np.get_include(), "../../make/", "../"],
    )
#define_macros = [ ('DISABLE_SSE2','0')]