# Copyright 2019 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup script."""

import os
import pathlib

from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()


setup(
    name="fed-boost",
    description="Federation learning boosting algorithm.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/lukewood/fed-boost",
    author="Abhijith-S-D",
    license="Apache License 2.0",
    install_requires=["packaging", "absl-py", "regex", "tensorflow-datasets"],
    python_requires=">=3.7",
    extras_require={
        "tests": [
            "flake8",
            "isort",
            "black[jupyter]",
            "pytest",
            "pycocotools",
            "tensorflow",
            "tensorflow_datasets",
            "matplotlib",
            "seaborn",
        ],
    },
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    packages=find_packages(exclude=("*_test.py",)),
    include_package_data=True,
)
