from setuptools import setup, find_packages

setup(
  name = 'q-transformer',
  packages = find_packages(exclude=[]),
  version = '0.1.2',
  license='MIT',
  description = 'Q-Transformer',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/q-transformer',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention mechanisms',
    'transformers',
    'q-learning'
  ],
  install_requires=[
    'accelerate',
    'beartype',
    'classifier-free-guidance-pytorch>=0.4.2',
    'einops>=0.7.0',
    'ema-pytorch>=0.3.1',
    'numpy',
    'torchtyping',
    'torch>=2.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
