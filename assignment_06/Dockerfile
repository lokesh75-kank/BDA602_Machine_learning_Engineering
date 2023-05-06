##My current ubuntu version is 22.04.2
##FROM ubuntu:22.04.2
##FROM Python:3.10
#FROM ubuntu:latest

FROM ubuntu:latest

# Seting a working directory for the app
WORKDIR /app

RUN mkdir ./sql_files
RUN chmod -R 777 ./sql_files

RUN mkdir ./sh_files
RUN chmod -R 777 ./sh_files

RUN mkdir ./output
RUN chmod -R 777 ./output
# Installing necessary packages and dependencies
RUN apt-get update && apt-get install -y \
    mariadb-client \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3-pip \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Installing Python packages using pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying app files
COPY ./baseball.sql ./sql_files/
COPY ./assignment_06/hw02.sql ./sql_files/
COPY ./assignment_06/baseball_db.sh ./sh_files/

# Making scripts executable
#chmod +x /app/baseball_db.sh /app/main.py

# Starting the app using the CMD command
#CMD ["/app/baseball_db.sh"]

RUN chmod +rwx ./sh_files/baseball_db.sh
CMD ["/bin/bash", "./sh_files/baseball_db.sh"]
