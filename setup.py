from setuptools import setup

with open('D:\\PythonProject\\TorchLoop\\requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='zephram',
    version='1.0',
    author='Xavier',
    author_email='zephramxu@gmail.com',
    url='https://github.com/ArdenteX/zephram',
    packages=['zephram'],
    install_requires=required
)

