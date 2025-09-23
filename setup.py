from setuptools import setup, find_packages

setup(
    name='Bereshit',
    version='0.1',
    packages=find_packages(),
    install_require=[
        'trimesh matplotlib',
        'moderngl',
        'pyserial',
        'keyboard',
        'moderngl_window',
        'pyrr',
        'mouse',
        'glfw',
        'open3d'
    ]
)
