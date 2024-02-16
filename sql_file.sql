/* this is just a test */

CREATE DATABASE my_database;


USE my_database;


CREATE TABLE customers (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  email VARCHAR(255) UNIQUE,
  phone_number VARCHAR(20)
);


SHOW TABLES;

DESCRIBE customers;