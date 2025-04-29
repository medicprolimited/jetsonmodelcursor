FROM dustynv/l4t-pytorch:r36.4.0

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY app.py .

# Expose the port
EXPOSE 8001

# Run the application
CMD ["python3", "app.py"]