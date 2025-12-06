from setuptools import find_packages, setup
from typing import List

HYPEN_E = "-e ."

def get_requirements(file_path: str) -> List[str]:
    '''
    It returns the list of requirements
    '''
    req = []
    with open(file_path) as file_obj:
        req = file_obj.readlines()
        req = [i.replace("\n", "") for i in req]
        if HYPEN_E in req:
            req.remove(HYPEN_E)
    return req


setup(
    name='mlproject',
    version='0.0.1',
    author='balaji',
    author_email='balajikundrapu@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
