from setuptools import setup, find_packages


setup(
    name="simpledl",
    version='0.0.1',
    author='Yongfei Yan',
    author_email='yongfeiyan@pku.edu.cn',
    url='https://github.com/YongfeiYan/simpledl',
    description='A simple toolkit for deep learning.',
    packages=find_packages(include=['simpledl/**', 'examples/*', 'scripts/*']),
    tests_require=['pytest'],
)

