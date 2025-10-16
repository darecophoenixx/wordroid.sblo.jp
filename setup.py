from distutils.core import setup

setup(
    name='wordroid.sblo.jp',
    authoer='Norio Tamada',
    url='https://github.com/darecophoenixx/wordroid.sblo.jp',
    version='0.00.02',
    package_dir={
        'feature_eng': 'lib/feature_eng',
        'keras_ex.gkernel': 'lib/keras_ex/gkernel',
        'keras_ex.HumanisticML': 'lib/keras_ex/HumanisticML',
        'som': 'lib/som',
        'egreedy': 'lib/egreedy'
        },
    packages=[
        'feature_eng',
        'keras_ex.gkernel',
        'keras_ex.HumanisticML',
        'som',
        'egreedy',
        ],
    install_requires=['gensim',],
)

