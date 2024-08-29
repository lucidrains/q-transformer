from setuptools import setup, find_packages

setup(
  name = 'q-transformer',
  packages = find_packages(exclude=[]),
  version = '0.2.0',
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
    'adam-atan2-pytorch>=0.0.12',
    'beartype',
    'classifier-free-guidance-pytorch>=0.6.10',
    'einops>=0.8.0',
    'ema-pytorch>=0.5.3',
    'jaxtyping',
    'numpy',
    'sentencepiece',
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
