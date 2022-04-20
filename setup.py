from setuptools import find_packages, setup
setup(
    name='flox',
    packages=find_packages(include=['flox']),
    version='0.1.0',
    description='Library for serverless Federated Learning experiments.',
    url='https://github.com/nikita-kotsehub/FLoX',
    author='Nikita Kotsehub',
    author_email='mykyta.kotsehub@gmail.com',
    license='MIT',
    install_requires=['numpy', 'funcx', 'parsl']
)