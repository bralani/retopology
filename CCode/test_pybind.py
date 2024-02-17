from setuptools import setup, Extension

# Compile *setCurvature.cpp* into a shared library 
setup(
    #...
    ext_modules=[Extension('setCurvature', ['setCurvature.cxx', 'Polyhedron.cxx', 'SvdSolve.cxx', 'PointTool.cxx', 'IDSet.cxx', 'Eigens.cxx'],),],
)
