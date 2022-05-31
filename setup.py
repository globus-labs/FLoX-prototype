from setuptools import find_packages, setup
setup(
    name='pyflox',
    packages=find_packages(include=['pyflox']),
    version='0.1.0',
    description='Library for serverless Federated Learning experiments.',
    url='https://github.com/nikita-kotsehub/FLoX',
    download_url='https://github.com/nikita-kotsehub/FLoX/archive/refs/tags/v0.1.1.tar.gz',
    author='Nikita Kotsehub',
    author_email='mykyta.kotsehub@gmail.com',
    license='MIT',
    install_requires=['numpy', 'funcx', 'parsl'],
    keywords = ['federated_learning', 'serverless', 'edge_devices'],
    classifiers=[
        'Development Status :: 3 - Alpha', 
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License', 
        'Programming Language :: Python ::   3',   
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
  ],
)