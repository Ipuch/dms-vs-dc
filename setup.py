from setuptools import setup

setup(
    name="robot_leg",
    version="0.1.0",
    author="Pariterre",
    packages=[
        "robot_leg",
        "robot_leg/models",
        "robot_leg/bioptim_plugin",
        "robot_leg/ocp",
    ],
    include_package_data=True,
    python_requires=">=3.7",
    zip_safe=False,
)