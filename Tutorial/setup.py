from distutils.core import setup


setup(name="Tutorial",
	version="1.02",
	description="Scoring profile optimization",
	author="Ivan Barrientos",
	author_email="ivbarrie@microsoft.com",
	packages=[
		"lib_utils",
		"lib_utils_async"
	],
	install_requires=[
        "numpy=1.20.2",
        "pandas=0.25.3",
		"optuna==0.19.0"
	],
)