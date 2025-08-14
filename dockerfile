# Use Python 3.11 (matching your local virtual environment)
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements file first (to optimize Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose Flask port
EXPOSE 5000

# Start Flask app using Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]

