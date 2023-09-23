from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess

def read_requirements():
    with open('requirements.txt', 'r') as file:
        return file.readlines()

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        subprocess.call(["git", "submodule", "update", "--init", "--recursive"])
        subprocess.call(["./install_submodules.sh"])

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        subprocess.call(["git", "submodule", "update", "--init", "--recursive"])
        subprocess.call(["./install_submodules.sh"])

setup(
    name='GLOMA_v2',
    version='0.1',
    packages=find_packages(),
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
    install_requires=read_requirements(),
)