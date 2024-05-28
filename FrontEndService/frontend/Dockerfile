# Use an official Python runtime as a parent image
FROM python:3.12

# Set the working directory in the container
WORKDIR /usr/src/app/

# Copy the current directory contents into the container at /usr/src/app
COPY frontend.py .

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.port=8501"]
