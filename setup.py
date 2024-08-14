from setuptools import setup, find_packages

setup(
    name="JXAutoML",  # Replace with your package name
    version="0.2.1",
    author="Lang Chen",
    author_email="ronchen6666@gmail.com",
    description="Advanced HyperParameter Tuning",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    # Replace with your GitHub repo
    url="https://github.com/TGChenZP/JXAutoML",
    packages=find_packages(),  # Automatically find all packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
