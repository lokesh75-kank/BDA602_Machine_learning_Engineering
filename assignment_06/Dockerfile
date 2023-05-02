#My current ubuntu version is 22.04.2
FROM ubuntu:22.04.2

ENV APP_HOME /app
WORKDIR $APP_HOME

RUN apt-get update \
    && apt-get install -y python3.10 mariadb-client \
    && rm -rf /var/lib/apt/lists/*

    #installs Python 3.10 and mariadb_client in my Docker image
    ## rm: command removes the package index files from the Docker image to reduce the size of the image.

ENV PYTHONPATH /

RUN mkdir ./result
RUN chmod +x ./baseball_db.sh
CMD ["/bin/sh", "-c", "./baseball_db.sh"]