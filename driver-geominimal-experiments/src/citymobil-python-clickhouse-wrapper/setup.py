from setuptools import setup

setup(
    name='citymobil_python_clickhouse_wrapper',
    version='0.2.1',
    description='Our ClickHouse wrapper for Python',
    url='git@gitlab.city-srv.ru:public_city/citymobil_python_clickhouse_wrapper.git',
    author='Aleksei Kozlov',
    author_email='a.kozlov@city-mobil.ru',
    license='No license. All rights reserved',
    packages=['citymobil_python_clickhouse_wrapper'],
    install_requires=[
        'newrelic',
        'pandas',
        'aiochclient',
        'nest_asyncio',
    ],
    zip_safe=False)
