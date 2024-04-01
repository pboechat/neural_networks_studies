from setuptools import setup, find_packages


def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f.readlines()]


setup(
    name='neural_network_studies',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires='>=3.6',
)
