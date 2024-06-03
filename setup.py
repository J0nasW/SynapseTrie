from setuptools import setup, find_packages

setup(
    name='SynapseTrie',
    version='0.2',
    packages=find_packages(),
    description='Efficient trie for storing and searching phrases.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Jonas Wilinski',
    author_email='jonas@wilinski.me',
    url='https://github.com/J0nasW/SynapseTrie',
    install_requires=[
        'pandas>=1.1.5',
        'nltk>=3.5',
        'scipy>=1.5.4',
        'tqdm>=4.56.0',
        'pyyaml>=5.4.1',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
    ],
)