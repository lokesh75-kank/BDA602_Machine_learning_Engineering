#!/bin/sh

set -e
#set -e to make the script exit immediately if any command fails.

# Wait for the database to be ready
echo "Waiting for database to be ready..."
while ! mysqladmin ping -h db -u admin -p1196 --silent; do
    sleep 2
done
echo "Database is ready!"
# Check if the database exists, and create it if it doesn't
if ! mysql -h db -u admin -p1196 -e 'use baseball;'; then
  echo "Creating baseball database..."
  mysql -h db -u admin -p1196 -e "create database baseball;"
  echo "Loading database..."
  mysql -h db -u admin -p1196 baseball < /app/res/baseball.sql
else
  echo "baseball database already exists"
fi

# Run the Python script
echo "Running Python script..."
python /app/main.py
echo "Python script execution done"

