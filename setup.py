from distutils.core import setup

setup(
    name='wordroid.sblo.jp',
    authoer='Norio Tamada',
    url='https://github.com/darecophoenixx/wordroid.sblo.jp',
    version='0.00.01',
    package_dir={
        'feature_eng': 'lib/feature_eng',
        'keras_ex.gkernel': 'lib/keras_ex/gkernel',
        'keras_ex.HumanisticML': 'lib/keras_ex/HumanisticML',
        'som': 'lib/som',
        },
    packages=[
        'feature_eng',
        'keras_ex.gkernel',
        'keras_ex.HumanisticML'
        'som',
        ],
    install_requires=['gensim',],
)

