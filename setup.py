from setuptools import setup, find_packages

setup(
  name = 'q-transformer',
  packages = find_packages(exclude=[]),
  version = '0.4.5',
  license = 'MIT',
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
    'adam-atan2-pytorch>=0.0.12',
    'beartype',
    'classifier-free-guidance-pytorch>=0.7.1',
    'einops>=0.8.0',
    'ema-pytorch>=0.5.3',
    'hl-gauss-pytorch>=0.1.17',
    'hyper-connections>=0.1.7',
    'jaxtyping',
    'numpy',
    'sentencepiece',
    'x-transformers>=2.0.2',
    'torch>=2.0'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
