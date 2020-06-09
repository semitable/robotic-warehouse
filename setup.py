from setuptools import setup, find_packages

setup(
    name="robotic_warehouse",
    version="0.0.1",
    description="Robotic Warehouse Environment for RL",
    author="Filippos Christianos",
    url="https://github.com/semitable/robotic-warehouse",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=[
        "numpy",
        "gym>=0.15",
        "pyglet",
        "networkx",
    ],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
