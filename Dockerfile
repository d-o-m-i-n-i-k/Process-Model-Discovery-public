FROM python:3

WORKDIR /opt/app

RUN apt-get update && apt-get install -y \
    graphviz xdg-utils

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . ./

CMD ["./run.sh"]