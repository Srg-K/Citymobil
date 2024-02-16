# we tried python:3.7-alpine, but pip packages, like pandas, can't be distributed precompiled and compile from c++ sources on Alpine. It takes hours. https://github.com/docker-library/python/issues/381#issuecomment-463880366
# this is trade-off between space (~500MB debian image vs ~200MB alpine image) and speed of container build time (~2 minutes on debian, forever on alpine)
FROM python:3.7-slim as base_container
WORKDIR /data
# we need gettext for envsubst; git for pip; cron for cron;
RUN apt update && apt install -y --no-install-recommends gettext-base git cron && rm -rf /var/lib/apt/lists/*
RUN pip install pipenv
COPY Pipfile Pipfile
COPY Pipfile.lock Pipfile.lock
RUN pipenv install
COPY . .

FROM base_container as tests
CMD ./wait-for-it.sh "local_mysql:3306" -- pipenv run python -m unittest discover

FROM base_container as linters
RUN pipenv install --dev
CMD pipenv run autopep8 . && pipenv run mypy . && pipenv run pylint -j 4 **/*.py

FROM base_container AS coverage
RUN pipenv install --dev
CMD ./wait-for-it.sh "local_mysql:3306" -- pipenv run coverage run --source . -m unittest discover && pipenv run coverage report

FROM base_container as pip_install
CMD ./wait-for-it.sh "local_mysql:3306" -- \
    pip install git+https://gitlab+deploy-token-83:UVXbZBQUGhLdTgxsxtxy@gitlab.city-srv.ru/marketplace/citymobil-python-mysql-wrapper.git@0.1.0#egg=citymobil_python_mysql_wrapper && \
    cd /tmp && cp /data/usage_example.py . && python usage_example.py
