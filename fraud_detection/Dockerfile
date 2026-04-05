# Use lightweight Python 3.11 image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (to leverage Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
# This includes the src directory and the crucial models/ directory
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the application (bound to 0.0.0.0 for external access)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
