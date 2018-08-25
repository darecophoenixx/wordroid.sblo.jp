from distutils.core import setup

setup(
    name='wordroid.sblo.jp',
    authoer='Norio Tamada',
    url='https://github.com/darecophoenixx/wordroid.sblo.jp',
    version='0.00.01',
    package_dir={'feature_eng': 'lib/feature_eng', 'keras_ex.gkernel': 'lib/keras_ex/gkernel',},
    packages=['feature_eng', 'keras_ex.gkernel',],
)

