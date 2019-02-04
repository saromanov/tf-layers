#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools
import textwrap
from pip.req import parse_requirements

version = "1.0"

def get_requirements():
    install_reqs = parse_requirements('requirements.txt', session='tf-layers')
    return [str(ir.req) for ir in install_reqs]

if __name__ == "__main__":
    setuptools.setup(
        name="tf-layers",
        version=version,
        description="Collection of neural network layers based on Tensorflow",
        author="Sergey Romanov",
        author_email="xxsmotur@gmail.com",
        long_description=textwrap.dedent("""\
            TODO
           """),
        packages=[
            "models",
        ],
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Environment :: Web Environment",
            "Intended Audience :: Developers",
            "License :: MIT",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Topic :: Software Development",
        ],
        python_requires=">=3.0",
        install_requires=get_requirements()
    )