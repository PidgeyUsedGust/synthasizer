from setuptools import setup, find_packages


setup(
    name="synthasizer",
    version="1.5.0",
    author="Gust Verbruggen",
    author_email="gust.verbruggen@kuleuven.be",
    description="Wrangling Tool",
    long_descripton="file: README.md",
    long_descripton_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["textdistance"],
    packages=find_packages(),
)
