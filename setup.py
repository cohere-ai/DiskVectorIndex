from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="DiskVectorIndex",
    version="0.0.2",
    author="Nils Reimers",
    author_email="nils@cohere.com",
    description="Efficient vector DB on large datasets from disk, using minimal memory.",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/cohere-ai/DiskVectorIndex",
    download_url="https://github.com/cohere-ai/DiskVectorIndex/",
    packages=find_packages(),
    install_requires=[
        'faiss-cpu==1.8.0',
        'numpy',
        'cohere>=5.5.4',
        'packaging',
        'tqdm',
        'indexed-zstd'
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="Vector Database"
)