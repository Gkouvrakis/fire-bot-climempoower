# Use an official Python runtime
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for geopandas/fiona/rasterio
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Upgrade pip and install Python libraries
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your script
COPY ondemand_fire_bot.py .

# Run the script
CMD ["python", "-u", "ondemand_fire_bot.py"]
