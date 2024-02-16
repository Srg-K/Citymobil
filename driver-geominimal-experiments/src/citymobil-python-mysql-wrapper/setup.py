from setuptools import setup

setup(
    name='citymobil_python_mysql_wrapper',
    version='0.1.0',
    description='Our MySql wrapper for Python',
    url='git@gitlab.city-srv.ru:marketplace/citymobil-python-mysql-wrapper.git',
    author='Aleksei Kozlov',
    author_email='a.kozlov@city-mobil.ru',
    license='No license. All rights reserved',
    packages=['citymobil_python_mysql_wrapper'],
    install_requires=[
        'newrelic',
        'pandas',
        'pymysql',
        'cryptography',
    ],
    zip_safe=False)
