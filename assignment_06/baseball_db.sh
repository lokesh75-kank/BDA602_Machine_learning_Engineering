#!/bin/sh

# Function to handle errors
handle_error() {
  echo "Error: $1" >&2
  exit 1
}

#set -e
#set -e to make the script exit immediately if any command fails.
# Wait for the database to be ready
echo "Waiting for database to be ready..."
while ! mysqladmin ping -h Lokesh_mariadb -u root -p1196 --silent; do
  sleep 15
done
echo "Database is ready!"

sleep 10

# Check if the database exists, and create it if it doesn't
if ! mysql -h Lokesh_mariadb -u root -p1196 -e 'use baseball;'; then
  echo "Creating baseball database..."
  mysql -h Lokesh_mariadb -u root -p1196 -e "create database baseball;" || handle_error "Failed to create baseball database"
  echo "Loading database..."
  mysql -h Lokesh_mariadb -u root -p1196 baseball < ./sql_files/baseball.sql || handle_error "Failed to load database"
else
  echo "baseball database already exists"
fi

# Run the Python script
#echo "Running Python script..."
##python3 /app/main.py
#echo "Python script execution done"

echo "Running hw_02_sql_script..."
mysql -h Lokesh_mariadb -u root -p1196 baseball < ./sql_files/hw02.sql || handle_error "Failed to execute SQL script"
echo "SQL script execution done"

echo "Exporting data ..."
mysql -h Lokesh_mariadb -u root -p1196 baseball -e 'SELECT * from rolling_100_partition where game_id = 12560;' > ./output/result.txt || handle_error "Failed to export data"
echo "Data exported to file ./output/result.txt"