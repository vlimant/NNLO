from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()

setup_args = dict(
    name='nnlo',
    version='0.0.5',
    entry_points = {
        'console_scripts': ['TrainingDriver=nnlo.driver.TrainingDriver:main',
            'GetData=nnlo.data.getdata:main',
            'CountEpoch=nnlo.util.count_epoch:main',
        ],
    },
    description='Distributed Machine Learning tool for High Performance Computing',
    long_description_content_type="text/markdown",
    long_description=README + '\n\n' + HISTORY,
    license='MIT',
    packages=find_packages(),
    author='NNLO team',
    author_email='rui.zhang@cern.ch',
    keywords=['Distributed Machine Learning', 'High Performance Computing', 'Hyperparameter optimisation'],
    url='https://github.com/chnzhangrui/NNLO',
    download_url='https://pypi.org/project/nnlo/'
)

install_requires = [
    'scikit-optimize',
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires, include_package_data=True)
