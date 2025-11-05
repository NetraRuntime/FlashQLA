import ast
import os
import re

from setuptools import find_packages, setup


def get_package_version():
    init_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flav2', '__init__.py')
    with open(init_file) as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    if version_match is None:
        raise RuntimeError(f"Could not find `__version__` in the file {init_file}")
    return ast.literal_eval(version_match.group(1))


setup(
    name='flash-linear-attention-v2',
    version=get_package_version(),
    description='Tilelang Ops for Gated Delta Rule',
    author='Chengruidong Zhang',
    author_email='chengruidong.zcrd@alibaba-inc.com',
    packages=find_packages(),
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.8',
        'tilelang==0.1.7'
    ]
)
