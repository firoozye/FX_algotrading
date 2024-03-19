# python -m pip install --upgrade setuptools

import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
    setuptools.setup(
        name='FX_algostrategy',
        version='0.1',
        packages=['utils', 'features', 'allocators'],
        author_email=  'firoozye@gmail.com',
        author='firoozye',
        description="<FX Algo_Strategy Setup.py package>",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="<https://github.com/authorname/templatepackage>",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Ap proved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.8',
        install_requires=['scipy', 'numpy', 'pandas', 'sklearn', 'matplotlib', 'xgboost', 'joblib'],  # Optional keyword
    )

# TODO: need version numbers
