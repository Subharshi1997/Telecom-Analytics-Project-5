-- MySQL initialization script
-- Creates the telecom_analytics database and grants permissions

CREATE DATABASE IF NOT EXISTS `telecom_analytics`
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

GRANT ALL PRIVILEGES ON `telecom_analytics`.* TO 'telecom_user'@'%';
FLUSH PRIVILEGES;

USE `telecom_analytics`;

-- user_satisfaction table will be created by the Python exporter (SQLAlchemy)
-- This script just ensures the database exists with correct permissions.

