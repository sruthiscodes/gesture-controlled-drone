from setuptools import setup
import os
from glob import glob

package_name = 'gesture_drone_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='Gesture controlled drone simulation',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gesture_drone_controller = gesture_drone_sim.gesture_drone_controller:main',
            'gesture_control = gesture_drone_sim.gesture_control:main',
        ],
    },
) 