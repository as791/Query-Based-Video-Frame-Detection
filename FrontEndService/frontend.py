import streamlit as st
import requests
from datetime import datetime, time
from PIL import Image
from io import BytesIO
import os

os.environ['TZ'] = 'UTC'


# Replace these with your actual API endpoints
UPLOAD_API_ENDPOINT = 'http://host.docker.internal:8080/v1/video/uploadVideo'
QUERY_API_ENDPOINT = 'http://host.docker.internal:8080/v1/video/search'

# Combine date and time into an epoch timestamp
def combine_datetime_to_epoch(date, time):
    combined_datetime = datetime.combine(date, time)
    return int(combined_datetime.timestamp())*1000

# Function to upload video
def upload_video(file):
    files = {'file': file}
    try:
        requests.post(UPLOAD_API_ENDPOINT, files=files)
    except:
        st.error('Failed to upload video')
        return None

# Function to query video
def query_video(start_time, end_time, text_prompt):
    params = {'startTime': start_time, 'endTime': end_time, 'query': text_prompt}
    try:
        response = requests.get(QUERY_API_ENDPOINT, params=params)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to query video: {e}")
        return None

# Function to generate curl command for the GET request
def generate_curl_command(endpoint, params):
    param_str = ' '.join([f'-d "{key}={value}"' for key, value in params.items()])
    return f'curl -G {param_str} "{endpoint}"'

# Download and display images from a list of URLs
def display_images_from_urls(urls):
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='Queried Frame', use_column_width=True)
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download image from {url}: {e}")
# Set up the Streamlit app
st.title("Video Upload and Query App")

# File uploader
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    if st.button('Upload'):
        upload_response = upload_video(uploaded_file)
        if upload_response:
            st.success(f"Uploaded: {upload_response}")

# Query inputs
start_date = st.date_input("Start Date")
start_time = st.time_input("Start Time", time(0, 0))
end_date = st.date_input("End Date")
end_time = st.time_input("End Time", time(0, 0))
text_prompt = st.text_input("Enter your query text")

if st.button('Query'):
    if start_date and start_time and end_date and end_time and text_prompt:
        start_epoch = combine_datetime_to_epoch(start_date, start_time)
        end_epoch = combine_datetime_to_epoch(end_date, end_time)
        st.write(f"Querying with start_time: {start_epoch}, end_time: {end_epoch}, text_prompt: {text_prompt}")
        
        # Generate and print the curl command
        params = {
            'startTime': start_epoch,
            'endTime': end_epoch,
            'query': text_prompt
        }
        curl_command = generate_curl_command(QUERY_API_ENDPOINT, params)
        st.code(curl_command, language='bash')
        
        query_response = query_video(start_epoch, end_epoch, text_prompt)
        if query_response:
            st.write("Query Results:")
            st.json(query_response)
            frame_urls = query_response.get("links")
            if frame_urls:
                display_images_from_urls(frame_urls)
            else:
                st.error("Frame URL not found in the response")
        else:
            st.error("Query failed")
    else:
        st.error("Please provide start date, start time, end date, end time, and text prompt")