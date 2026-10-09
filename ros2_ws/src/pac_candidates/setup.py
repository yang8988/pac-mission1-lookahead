from setuptools import setup

package_name = "pac_candidates"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="taehyeon",
    maintainer_email="taehyeon@example.com",
    description="AHEAD stages 5-1 candidate generation and 5-2 hard mask",
    license="Proprietary",
)
