# Use the official CUDA base image from NVIDIA with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables for CUDA and cuDNN
ENV CUDA_VERSION=11.8
ENV CUDNN_VERSION=8
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# Set the working directory in the container
WORKDIR /app

# Copy the web_dev directory contents into the container at /app
COPY ./web_dev /app

# Copy the database directory to the parent directory of /app
COPY ./database /database

RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    python3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install the CUDA toolkit and verify installation
RUN apt-get update && apt-get install -y cuda-toolkit-11-8 && rm -rf /var/lib/apt/lists/* && nvcc --version

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html

# Install python dependencies
RUN pip3 install --no-cache-dir Flask
RUN pip3 install --no-cache-dir transformers
RUN pip3 install --no-cache-dir accelerate
RUN pip3 install --no-cache-dir langchain
RUN pip3 install --no-cache-dir langchain_community
RUN pip3 install --no-cache-dir langchain_chroma
RUN pip3 install --no-cache-dir langchain_pinecone
RUN pip3 install --no-cache-dir langchain-huggingface
RUN pip3 install --no-cache-dir huggingface-hub
RUN pip3 install --no-cache-dir sentence-transformers
RUN pip3 install --no-cache-dir optimum
RUN pip3 install --no-cache-dir auto-gptq[cuda]
RUN pip3 install --no-cache-dir uwsgi

# Create a non-root user and group if they do not already exist
RUN groupadd -r appuser || true && useradd -r -g appuser -d /app -s /sbin/nologin appuser || true

# Create a directory for uWSGI logs and change ownership of the application directory to the non-root user
RUN mkdir -p /var/log/uwsgi && chown -R appuser:appuser /var/log/uwsgi /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/scripts && chown -R appuser:appuser /app/scripts && chmod -R 755 /app/scripts

# Copy the uwsgi.ini file into the container
COPY ./web_dev/uwsgi.ini /app/uwsgi.ini

# Expose the port the app runs on
EXPOSE 8123

# Define environment variable
ENV NAME World

# Switch to the non-root user
USER appuser

# Run uwsgi when the container launches
CMD ["uwsgi", "--ini", "uwsgi.ini"]
