import logging
import os
import sys

# Define the desired log folder and file name
log_folder = os.path.join(os.path.dirname(__file__), '../../logging')
log_folder = os.path.abspath(log_folder)
log_file = os.path.join(log_folder, "explora.log")

# Create the log folder if it doesn't exist
os.makedirs(log_folder, exist_ok=True)

# Get the root logger instance (so all modules inherit this config)
log = logging.getLogger()
log.setLevel(logging.DEBUG)  # Set the logging level as needed

# Remove all handlers associated with the root logger (avoid duplicate logs)
for handler in log.handlers[:]:
    log.removeHandler(handler)

# Create a FileHandler to write logs to the specified file
file_handler = logging.FileHandler(log_file)

# Create a Formatter to define the log message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set the formatter for the file handler
file_handler.setFormatter(formatter)

# Add the file handler to the logger
log.addHandler(file_handler)

# Optional: Also log to stdout for interactive feedback
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)

# Example usage (remove after importing in your main entrypoint):
# log.debug("This is a debug message that will go to the file and console.")
# log.info("Application started and this will also be in the file and console.")
# log.warning("A warning occurred, check the log file.")
# log.error("An error was encountered and logged.")
# log.critical("A critical error! See the log file for details.")
