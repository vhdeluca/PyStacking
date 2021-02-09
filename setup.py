from setuptools import setup, find_packages


setup(name='pystacking',
      version='0.1.0',
      description='Python Machine Learning Stacking Maker',
      author='Vitor Hugo Medeiros De Luca',
      author_email='vitordeluca@gmail.com',
      license='MIT',
      packages=find_packages(),
      python_requires=">=3.5",
      tests_require=['pytest'])
