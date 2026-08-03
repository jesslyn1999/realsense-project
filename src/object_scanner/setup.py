from glob import glob

from setuptools import find_packages, setup


package_name = "object_scanner"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        (
            "share/" + package_name + "/resource",
            glob("resource/*.json"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jess",
    maintainer_email="jess@example.com",
    description="Filter, transform, and record colored RealSense point clouds.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "scanner_node = object_scanner.scanner_node:main",
        ],
    },
)
