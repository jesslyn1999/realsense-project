from glob import glob

from setuptools import find_packages, setup


package_name = "object_scanner_web"


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
            "share/" + package_name + "/templates",
            glob("templates/*"),
        ),
        (
            "share/" + package_name + "/static",
            glob("static/*"),
        ),
        (
            "share/" + package_name + "/repair",
            ["../../resources/pipe-testing08-repair(3).stl"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jess",
    maintainer_email="jess@example.com",
    description="Flask controls and Three.js visualization for object_scanner.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "web_server = object_scanner_web.web_server:main",
        ],
    },
)
