FROM python:3.7
WORKDIR /app

COPY requirements.txt /app
RUN apt-get update
RUN pip install -r ./requirements.txt

COPY app.py /app
COPY mobilenet_0.835.h5 /app
COPY data_generator_sx3.py /app
CMD ["python", "app.py"]~
