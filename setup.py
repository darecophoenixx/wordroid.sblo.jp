from distutils.core import setup

setup(
    name='wordroid.sblo.jp',
    version='0.00.01',
    package_dir={'feature_eng': 'lib/feature_eng', 'keras_ex.GaussianKernel': 'lib/keras_ex/GaussianKernel',},
    packages=['feature_eng', 'keras_ex.GaussianKernel',],
)
