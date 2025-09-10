FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# Install system dependencies required for various libraries
RUN apt-get update && apt-get install -y \
    ffmpeg build-essential git libgl1-mesa-glx libsm6 libxext6 cmake-data pkg-config libhdf5-dev nano && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY ./requirements.txt /app/requirements.txt

# Install Python dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY ./app /app/app

# Expose the port the API will run on
EXPOSE 8002

# The command to run the application using uvicorn
# --host 0.0.0.0 makes the server accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]