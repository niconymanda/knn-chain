from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'knn_chain',
        ['knn_chain.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
    ),
]

setup(
    name='knn_chain',
    version='0.1',
    ext_modules=ext_modules,
)
