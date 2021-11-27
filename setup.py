from setuptools import setup, find_packages

setup(
    name="torchmdn",
    py_modules=["torchmdn"],
    install_requires=["torch"],
    packages=find_packages(),
)
