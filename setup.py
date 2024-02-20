from setuptools import setup, Extension

setup(
    #...
    ext_modules=[Extension('setCurvature', ['crest_lines/setCurvature.cxx', 'crest_lines/Polyhedron.cxx', 'crest_lines/SvdSolve.cxx', 'crest_lines/PointTool.cxx', 'crest_lines/IDSet.cxx', 'crest_lines/Eigens.cxx'],),],
)
