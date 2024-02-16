FROM python:3
ADD rfm_test.py /
ADD sql_data_rfm.py /
ADD config.py /
ADD requirements.txt /
RUN pip3 install -r /requirements.txt
CMD [ "python", "./rfm_test.py" ]