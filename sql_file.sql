/* this is just a test */
-- Create a database named "my_database"
CREATE DATABASE my_database;

-- Use the newly created database
USE my_database;

-- Create a table named "customers" with four columns
CREATE TABLE customers (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  email VARCHAR(255) UNIQUE,
  phone_number VARCHAR(20)
);

-- Show the created table
SHOW TABLES;

-- Describe the table structure
DESCRIBE customers;