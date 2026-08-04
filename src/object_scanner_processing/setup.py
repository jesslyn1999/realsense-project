from setuptools import find_packages, setup


package_name = "object_scanner_processing"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "open3d-cpu==0.19.0",
        "scikit-learn>=1.6.0",
        "scipy>=1.15.0",
        "setuptools",
    ],
    zip_safe=True,
    maintainer="jess",
    maintainer_email="jess@example.com",
    description=(
        "Hardware-independent refinement of recorded object-scanner point clouds."
    ),
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
)
