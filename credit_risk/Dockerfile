# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY credit_risk/src /app/src
COPY credit_risk/models /app/models
COPY credit_risk/data/processed/preprocessor.joblib /app/data/processed/

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run uvicorn when the container launches
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
