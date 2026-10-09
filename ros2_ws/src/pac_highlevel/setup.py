from glob import glob

from setuptools import setup

package_name = "pac_highlevel"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/models", glob("models/*")),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="taehyeon",
    maintainer_email="taehyeon@example.com",
    description="AHEAD stage 4 high-level action selection (rules + MaskablePPO)",
    license="Proprietary",
)
