FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Build the frontend
WORKDIR /app/react-frontend
RUN apt-get update && apt-get install -y nodejs npm
RUN npm install && npm run build

# Return to app directory
WORKDIR /app

# Port that the container will listen on
EXPOSE 10000

# Set environment variable
ENV PORT=10000

# Command to run the application
CMD gunicorn --bind 0.0.0.0:$PORT app:app 