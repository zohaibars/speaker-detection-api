from fastapi import FastAPI, UploadFile, Depends, HTTPException, Header

import moviepy.editor as mp
import os
import concurrent.futures
import asyncio
import shutil
from fastapi.responses import JSONResponse
from test import main_part
from utils import *
app = FastAPI()

# Define a directory to store uploaded videos and audio
upload_dir = "uploads"
# Function to clear the upload directory
def clear_upload_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
clear_upload_directory(upload_dir)

user_api_keys = {
    "user1": "apikey1",
    "user2": "apikey2",
    # Add more users and their API keys as needed
}

def process_video(file: UploadFile):
    try:
        # Read the uploaded video file into memory
        file_content = file.file.read()
        file_name, file_extension = os.path.splitext(file.filename)
      # Create a folder for the file inside the "uploads" directory
        file_folder = os.path.join(upload_dir, file_name)
        os.makedirs(file_folder, exist_ok=True)

        # Save the uploaded file in the file folder
        file_path = os.path.join(file_folder, file.filename)
        with open(file_path, "wb") as temp_file:
            temp_file.write(file_content)

         # Check if the file has a video extension
        video_extensions = [".mp4", ".avi", ".mov", ".wmv", ".mkv", ".flv"]
        if any(file_extension.lower().endswith(ext) for ext in video_extensions):
        # Save the audio clip in the same directory as the video
            audio_file_path = os.path.join(file_folder, f"{file_name}.wav")
            if not os.path.exists(audio_file_path):
                # Use moviepy to process the saved video file
                video_clip = mp.VideoFileClip(file_path)

                # Extract audio from the video in memory
                audio_clip = video_clip.audio

                audio_clip.write_audiofile(audio_file_path)
                video_clip.close()
                # Close the audio clip
                audio_clip.close()
            else:
                pass
        else:
            audio_file_path=file_path
        result=main_part(audio_file_path)
        shutil.rmtree(file_folder, ignore_errors=True)
        # Return as dictionary with line-by-line content
        return result

    

    except Exception as e:
        return JSONResponse(content={"error": f"Internal Error {str(e)}"}, status_code=500)
# Dependency to validate the API key
async def get_api_key(api_key: str = Header(None, convert_underscores=False)):
    if api_key not in user_api_keys.values():
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


@app.post("/Speaker_Recognition/")
async def Speaker_Recognition_endpoint(
    video_file: UploadFile,
    api_key: str = Depends(get_api_key),  # Require API key for this route

):
    # Create a new thread for processing each user's video
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: process_video(video_file)
        )
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="192.168.18.164", port=8011,reload=True)
    
# run command in cmd 
# uvicorn app:app --host 0.0.0.0 --port 2009 --reload