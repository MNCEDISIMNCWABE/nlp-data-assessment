# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt wordnet stopwords

# Copy the rest of the application
COPY . /app/

# Run model_training.py when the container launches
CMD ["python", "model_training.py"]