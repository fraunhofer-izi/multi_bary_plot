import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="multi_bary_plot",
    version="1.0",
    author="Dominik Otto",
    author_email="dominik.otto@gmail.com",
    description="Plots n-dimensional data into 2-d barycentric coordinates.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/katosh/multi_bary_plot",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'multiprocess',
        'tqdm',
        'matplotlib',
        'scipy',
    ],
)