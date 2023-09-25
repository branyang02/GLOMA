# from setuptools import setup, find_packages
# from setuptools.command.develop import develop
# from setuptools.command.install import install
# import subprocess

# def read_requirements():
#     with open('requirements.txt', 'r') as file:
#         return file.readlines()

# class PostDevelopCommand(develop):
#     """Post-installation for development mode."""
#     def run(self):
#         develop.run(self)
#         subprocess.call(["git", "submodule", "update", "--init", "--recursive"])
#         subprocess.call(["./install_submodules.sh"])

# class PostInstallCommand(install):
#     """Post-installation for installation mode."""
#     def run(self):
#         install.run(self)
#         subprocess.call(["git", "submodule", "update", "--init", "--recursive"])
#         subprocess.call(["./install_submodules.sh"])

# setup(
#     name='GLOMA_v1',
#     version='0.1',
#     packages=find_packages(),
#     cmdclass={
#         'develop': PostDevelopCommand,
#         'install': PostInstallCommand,
#     },
#     install_requires=read_requirements(),
# )



from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess
import os

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r') as file:
        return file.readlines()

# Update and initialize git submodules
def update_submodules():
    try:
        subprocess.run(["git", "submodule", "update", "--init", "--recursive"], check=True)
    except subprocess.CalledProcessError:
        print("Error updating git submodules.")
        exit(1)

# Install SAM
def install_sam():
    try:
        subprocess.run(["python", "-m", "pip", "install", "-e", "submodules/Grounded-Segment-Anything/segment_anything"], check=True)
    except subprocess.CalledProcessError:
        print("Error installing segment_anything.")
        exit(1)

def install_GroundingDINO():
    try:
        subprocess.run(["python", "-m", "pip", "install", "-e", "submodules/Grounded-Segment-Anything/GroundingDINO"], check=True)    
    except subprocess.CalledProcessError:
        print("Error installing GroundingDINO.")
        exit(1)


# Check environment variables
AM_I_DOCKER = os.environ.get("AM_I_DOCKER", "False")
BUILD_WITH_CUDA = os.environ.get("BUILD_WITH_CUDA", "True")
CUDA_HOME = os.environ.get("CUDA_HOME")

if AM_I_DOCKER == "True":
    print("Cannot build in Docker container")
    exit(1)

if BUILD_WITH_CUDA == "False":
    print("Cannot build without CUDA")
    exit(1)

if CUDA_HOME is None:
    print("CUDA_HOME environment variable is not set.")
    exit(1)

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        update_submodules()
        install_sam()
        install_GroundingDINO()
        develop.run(self)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        update_submodules()
        install_sam()
        install_GroundingDINO()
        install.run(self)

setup(
    name='GLOMA_v1',
    version='0.1',
    packages=find_packages(),
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
    install_requires=read_requirements(),
)
