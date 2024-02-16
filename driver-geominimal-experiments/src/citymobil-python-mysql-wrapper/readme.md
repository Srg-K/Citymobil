[![pipeline status](http://gitlab.city-srv.ru/marketplace/citymobil-python-mysql-wrapper/badges/master/pipeline.svg)](http://gitlab.city-srv.ru/marketplace/citymobil-python-mysql-wrapper/commits/master)
[![coverage report](http://gitlab.city-srv.ru/marketplace/citymobil-python-mysql-wrapper/badges/master/coverage.svg)](http://gitlab.city-srv.ru/marketplace/citymobil-python-mysql-wrapper/commits/master)

# pip пакет с нашей оберткой для mysql 
## про что проект?
Основное - добавлены методы для работы с pandas dataframe. Основан на pymysql

# инструкция по форку
* нажать в гитлабе кнопку fork на главной странице репо
    * критерий к имени форка `must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')`
    * он вырастает из ограничений имен ресурсов кубернетес, а k8s deploy называется как проект
* форк должен произойти в приватную группу проектов с именем вашего пользователя
* зайти в форкнутый репо и переименовать его (settings/general/advanced/rename repository) - менять имя и url
* сделать трансфер проекта обратно в группу marketplace (settings/general/advanced/transfer project)
* изменить описание проекта (settings/general/General project/Project description (optional))
* обновление `readme.md`
    * изменить первый заголовок 
    * актуализировать секцию "про что проект?" (второй параграф можно оставить, но прочитать перед этим)
    * budges билда и покрытия 
        * build budge (settings/ci\cd/general pipelines/Pipeline status/Markdown) 
        * coverage budge (settings/ci\cd/general pipelines/Coverage report/Markdown) - в поле Test coverage parsing нужно вставить регулярку: `TOTAL\s+\d+\s+\d+\s+(\d+\%)`
    * `newrelic links:`, `## логи` `## дашборды kubernetes` выставить в туду
* удалить лишние параметры из конфига и юнит-теста на конфиг (папка `config`). Сам yml править в файлах `config.yml` и `config.yml.dist`
* обновить в `newrelic.ini` все вхождения `app_name` (лучше ставить имя проекта из гитлаба)
* удалить `prime_numbers_finder` пакет и его использование в `main.py`
* **[закончи эту секцию перед первым коммитом!!!]** деплой в кубы:
    * внимательно посмотреть в `.k8s/deployment.yaml.dist`, но там ничего менять не нужно. Просто чтобы быть в курсе.
    * переменные в config
        * в `config.yml` для тестов на локалке и локальном docker 
        * на пайплайнах конфиг генерируется через утилиту `envsubst` из переменных окружения и файла `config.yml.dist` подменяя `config.yml` из репозитория.
         поэтому значения еще и в `config.yml.dist`, при чем если значение конфига отличается в деве и проде в значении нужно указать через переменную окружения, откуда возьмется значение по аналогии с переменной `NEWRELIC_ENVIRONMENT`.
            * переменные окружения в пайплайнах выставляются в (settings/ci\cd settings/environment variables)
                * пример установленных переменных в гитлабе можно посмотреть здесь https://gitlab.city-srv.ru/marketplace/python_cron_microservice_template/settings/ci_cd `Environment variables`
                * создать `Deploy Token` (settings/repository/deploy tokens) c именем `deploy_token_for_k8s` с пустым expires at, с галочкой на read registry
                    * значение password показывается только один раз. Его и username нужно перенести в переменные окружения форкнутого репозитория (settings/ci\cd settings/environment variables) DOCKER_REGISTRY_USERNAME и DOCKER_REGISTRY_PASSWORD
                    * они используются для создания k8s secret через который kubectl сходит в docker registry за образом контейнера   
                * из репо шаблона нужно перенести все существующие там переменные окружения в форкнутый репозиторий, кроме DOCKER_REGISTRY_USERNAME и DOCKER_REGISTRY_PASSWORD (см. выше что с ними делать) 
            * применяются при dev и prod билдах контейнера - в `.gitlab-ci.yml` в джобах `build_docker_dev` и `build_docker_prod` по аналогии с переменной `NEWRELIC_ENVIRONMENT`
                * обратите внимание что паттерн такой `VAL:$VAL_DEV`, где `$VAL_DEV` объявлена в гитлабе
            * передача перменной окружения в контейнер в джобе `.build_docker_template`, пример `--build-arg NEWRELIC_ENVIRONMENT=$NEWRELIC_ENVIRONMENT`
                * обратите внимание что паттерн такой `--build-arg VAL=$VAL`
            * в файле `Dockerfile` принятие аргумента по аналогии с `ARG NEWRELIC_ENVIRONMENT`
                * именно в `Dockerfile` содержится вызов `envsubst`
* опционально интегрировать jira cloud (секция "интеграция с jira")
* пример форка по инструкции - https://gitlab.city-srv.ru/marketplace/polygon_coefficient_model_assigner

# инструкция по подтягиванию новых изменений от родителя форка (от репо шаблона к дочернему проекту)
* в дочернем проекте добавляем remote `template_upstream`
    ```
    git remote add template_upstream git@gitlab.city-srv.ru:marketplace/python_cron_microservice_template.git
    ```
* в дочернем проекте фетчим изменения от родителя
    ```
    git fetch template_upstream
    ```
* мержим мастер родителя в мастер форкнутого ребенка
    ```
    # репо ребенка
    git checkout master
    git merge template_upstream/master
    # решаем конфликты, коммитим
    ```
    
# кандидатская диссертация на тему крона в докере:
При запуске из контейнера запускается процесс крона, который по расписанию из файла `my_cron` запускает `python main.py`
Мы используем перенаправление stdout и stderr вывода с крона в дескрипторы процесса 1 чтобы захватить логи, недопуская форка крона.
Недопускать форк крона нам нужно на случай если он упадет. Когда он pid 1 - оркестратор перезапустит контейнер,
но если крон упадет в форкнутом состоянии - то мы об этом не узнаем. 
А так мы добились того что крон - это пид 1, а вывод кронов идет в его stdout/stderr.
(мат часть кодов выходов - https://stackoverflow.com/questions/37458287/how-to-run-a-cron-job-inside-a-docker-container)  

# что можно сделать с докером
```
docker-compose up --build tests
docker-compose up --build linters
docker-compose up --build coverage
docker-compose up --build pip_install
docker-compose up --build local_mysql
```

# действия с кодом на локальной машине
* перед любыми командами нужно войти в pipenv shell
```shell script
pipenv shell
```
* убедиться что все зависимости установлены
```shell script
pipenv install
pipenv install --dev
```
* запуск
```shell script
python main.py
```
* прогнать форматер, линтеры и тесты
    ```
    clear && autopep8 . && mypy . && pylint -j 4 **/*.py && python3 -m unittest discover && echo ok || echo fail
    ```
     * autopep8 - форматирует сорцы на месте в соответствии с pep8
        * заигнорить строчку  (коммент в конце строки)
        ```
        # noqa
        ```
     * mypy - статическая проверка строгой типизации
     * pylint - линтер
        * расшифровка pylint кода ошибки (если из мэсэджа непонятно)
        ```
        pylint --help-msg=C6409
        ```
        * отключить для строчки (коммент в конце строки)
        ```python
        # pylint: disable=unused-import
        ```
     * python3 -m unittest discover - стандартный лаунчер стадартных питон тестов
* coverage:<br>
    игноры в файле `.coveragerc`<br>
    запуск:<br>
    ```
    clear && coverage run --source . -m unittest discover && pipenv run coverage report
    ```
    посмотреть **html** репорт(строчки таблицы кликабельны):
    ```
    clear && coverage run --source . -m unittest discover && pipenv run coverage report && coverage html && open ./htmlcov/index.html
    ```

# прочее полезное
## pipenv
дропнуть текущее состояние pipenv
```shell script
rm -rf `pipenv --venv`
pipenv install
```

## newrelic 
Настройки релика лежат в [файле newrelic.ini](newrelic.ini)
Ньюрелик инициализируется при старте скрипта - занимает ~5-10 секунд

от дефолта отличаются:
* [newrelic:production] - переопределение параметров для дев/прод среды
* [newrelic:test]
* license_key
* app_name
* startup_timeout - по дефолту релик инициализируется лениво при первом использовании, нужно дать явно время на синхронную инициализацию, чтобы не терять первую транзакцию из-за того что инициализация не успела завершиться
* shutdown_timeout - по дефолту релик отсылает раз в минуту данные и при смерти процесса ничего не отсылает, нужно явно дать время на отсыл данных при смерти процесса

newrelic links:
* dev
  * TODO
* prod
  * TODO
  * alerts
    * TODO
  * dashboard
    * TODO

## логи
* dev 
    * TODO
* prod
    * TODO

### Как добавить логи в новом проекте?
* есть описание устройства логов: https://www.notion.so/citymobilmrg/k8s-ELK-elastic-logstash-kibana-49a5058c627f443e916246665dfa73bb
* создать задачу в редмайне в проекте SysOps и отдать админам (предпочтительно Александру Аксеновскому)
    * до перенаправления логов приложения в отдельный индекс на дев и прод кибанах доступны из индекса `logstash*`, но нужно фильтроваться от других приложений, например по полю `kubernetes.pod.name` или другому, который подойдет лучше

## дашборды kubernetes
* dev
    * http://kubeworker-dev1.stl.msk2.city-srv.ru/#!namespace/marketplace
* prod
    * http://kubeworker-prod1.stl.msk2.city-srv.ru/#!namespace/marketplace
    
## как устроен деплой в k8s
* делаем коммит в master бранч
* gitlab pipeline начинает исполнение инструкций из `.gitlab-ci.yml`
* gitlab джобы build_docker_* билдят docker image с тэгами `<git_commit_hash>[_DEV|_PROD]`
    * контейнеры отличаются значениями переменных окружения, которые им передаются. Например, разные среды запуска NewRelic или базы данных
* билд каждого контейнера идет по инструкциям из `Dockerfile`
    * `Dockerfile` написан в стиле "multi-stage", т.е. содержит несколько точек входа CMD в зависимости от таргета.
    * во время билда контейнера контейнера при наличии переменной окружения `$APP_ENV` из `config.yml.dist` генерируется `config.yml` подробнее в `# инструкция по форку`
* если контейнер удалось сбилдить, он пушится в docker registry который принадлежит проекту в гитлабе
* в джобах `deploy_k8s_*`
    * из файла `.k8s/deployment.yaml.dist` генерируется `.k8s/deployment.yaml`, это инструкция дле деплоя в кубы
    * во время деплой kubectl скачивает образ контейнера из docker registry и отправляет его в кластер. про авторизацию kubectl в docker registry описано в `# инструкция по форку`
    * конфиг уже находится в контейнере т.к. генерировался на стадии билда контейнера, перед его размещением в registry

## интеграция с jira
мануал (секция для клауд) - https://docs.gitlab.com/ee/user/project/integrations/jira.html#configuring-gitlab

в джира клауд https://id.atlassian.com/manage/api-tokens создать api token. 

в гитлабе в проекте (settings/integrations/jira) указать:
* Web URL: https://city-mobil.atlassian.net/
* Username or Email: a.kozlov@city-mobil.ru
* Enter new password or api token: хранится у Алексея Козлова в тайне, передается наследнику только перед смертью
* остальные поля оставить пустыми
* нажать save 

После этого в комментарии коммита указать идентификатор таска, например: "[SURGE-1] readme update" - гитлаб отправит жире идентификатор коммита и в джира таске SURGE-1 появится коммент от юзера, кто сгенерил ключ интеграции по API.  
