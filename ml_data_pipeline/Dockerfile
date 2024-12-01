# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified by Poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-cache
RUN apt-get update && apt-get install -y git

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run main.py when the container launches
CMD ["python", "src/ml_data_pipeline/main.py", "config/config.yaml"]
