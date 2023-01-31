from setuptools import setup

setup(
    name="transcriptions",
    version="0.2.0",
    author="Ipuch",
    packages=[
        "transcriptions",
        "transcriptions/models",
        "transcriptions/bioptim_plugin",
        "transcriptions/ocp",
    ],
    include_package_data=True,
    python_requires=">=3.10",
    zip_safe=False,
)
