from setuptools import setup
import os

def readme():
    with open('README.md') as f:
        return f.read()

here = os.path.dirname(os.path.abspath(__file__))
version_ns = {}
with open(os.path.join(here, 'pygasflow', '_version.py')) as f:
    exec (f.read(), {}, version_ns)

setup(
    name = 'pygasflow',
    version = version_ns["__version__"],
    description = 'Ideal Gasdynamics utilities for Python 3.6+',
    long_description = readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.9',
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    keywords='gasdynamic shockwave fanno rayleigh isentropic flow perfect gas',
    url = 'https://github.com/Davide-sd/pygasflow',
    author = 'Davide Sandona',
    author_email = 'sandona.davide@gmail.com',
    license='GNU GPL v3',
    packages = [
        'pygasflow',
        'pygasflow.atd',
        'pygasflow.nozzles',
        'pygasflow.solvers',
        'pygasflow.utils',
    ],
    include_package_data=True,
    zip_safe = False,
    install_requires = [
        "numpy",
        "scipy",
        "matplotlib",
        "pandas"
        # "cantera"
    ]
)
