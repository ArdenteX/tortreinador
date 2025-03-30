from setuptools import setup

setup(
    name='tortreinador',
    version='0.1.8',
    author='Xavier',
    author_email='zephramxu@gmail.com',
    url='https://github.com/ArdenteX/tortreinador',
    packages=['tortreinador', 'tortreinador.models', 'tortreinador.utils'],
    install_requires=[
        'pandas>=1.4.2',
        'seaborn>=0.11.2',
        'matplotlib>=3.5.1',
        'scikit-learn>=1.0.2',
        'tqdm>=4.64.0',
        'tensorboardx>=2.6',
        'numpy>=1.21.5',
        'setuptools>=61.2.0'
    ]
)

