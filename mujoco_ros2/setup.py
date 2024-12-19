from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mujoco_ros2'

def recursive_glob(directory, prefix=""):
    """Recursively gather all files in a directory, maintaining relative paths."""
    paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, prefix) if prefix else abs_path
            paths.append((os.path.join('share', package_name, root), [abs_path]))
    return paths

# Collect all files in the 'meshes' directory and subdirectories
meshes_files = recursive_glob('models/meshes', prefix='models')

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'models'), glob('models/*.*')),
        *meshes_files,  # Add all files from the meshes directory
        (os.path.join('share', package_name, 'config'), glob('config/*.*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kimyeonguk',
    maintainer_email='kimyeonguk@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={           
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'mujoco_sim = mujoco_ros2.mujoco_sim_node:main',
            'base_controller = mujoco_ros2.base_controller:main',
        ],
    },
)
