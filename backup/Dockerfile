# Use an official Python runtime as a base image
FROM python:3.10-slim
 
# Set the working directory in the container
WORKDIR /app
 
# Copy the requirements file to the working directory
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 git wget && \
    rm -rf /var/lib/apt/lists/*

# Create directories for checkpoints
RUN mkdir -p checkpoints

# Download the checkpoint file
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -P checkpoints

# Install dependencies
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt 
 
# Install necessary system dependencies
RUN apt-get update && apt-get install -y git
 
# Clone the Segment Anything repository
RUN git clone https://github.com/facebookresearch/segment-anything-2.git
 
# Change working directory to the cloned repository
WORKDIR /app/segment-anything-2
 
# Install the required Python packages
RUN pip install -e . -q
 
# Build the extension in place
RUN python setup.py build_ext --inplace
 
# Change working directory to the cloned repository
WORKDIR /app/
 
# Copy the FastAPI application code, including helper files, to the container
COPY . .
 
# Expose the port FastAPI runs on
EXPOSE 8000
 
# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
