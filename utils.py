from moviepy.editor import VideoFileClip
import os
import shutil
import json
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence


def video2wav(filename):
    wav_file_path = filename.split(".")[0] + '.wav'
    video_clip = VideoFileClip(filename)
    audio_clip = video_clip.audio
    # Export the audio as a WAV file
    audio_clip.write_audiofile(wav_file_path)
    video_clip.close()
    audio_clip.close()
    return wav_file_path


def save_audio_file(person_name, audio_file_path):
    # Create the 'audios' folder if it doesn't exist
    if not os.path.exists('audios'):
        os.mkdir('audios')

    # Create a folder with the person's name if it doesn't exist
    person_folder = os.path.join('audios', person_name)
    if not os.path.exists(person_folder):
        os.mkdir(person_folder)

    # Get the list of existing files in the person's folder
    existing_files = os.listdir(person_folder)

    # Determine the new file name
    new_file_name = str(len(existing_files) + 1) + "." +audio_file_path.split(".")[-1]

    # Construct the full path to the new audio file
    new_audio_file_path = os.path.join(person_folder, f'{new_file_name}')

    # Copy the audio file to the new location with the unique name
    shutil.copy(audio_file_path, new_audio_file_path)
    return new_audio_file_path


def save_embeddings(speakers=None):
    data = []  # This is the new data you want to append
    
    # Load the existing data from the JSON file if it exists
    try:
        with open("speaker_embeddings.json", "r") as json_file:
            existing_data = json.load(json_file)
        data.extend(existing_data)
    except FileNotFoundError:
        pass
    
    for speaker_info in speakers:
        embedding = speaker_info["embeddings"]
        label_name = speaker_info["label_name"]
        audio_file = speaker_info["audio_file"]
    
        # Store the information in a dictionary
        entry = {
            "label_name": label_name,
            "audio_embeddings": embedding.tolist(),
            "audio_file": audio_file,
        }
        data.append(entry)
    
    # Save the updated data to the JSON file
    with open("speaker_embeddings.json", "w") as json_file:
        json.dump(data, json_file)
    
    # print("Data appended to speaker_embeddings.json")


def load_embeddings():
    # Initialize lists to store the loaded data
    label_names = []
    embeddings = []
    try:
        # Load the data from the JSON file
        with open("speaker_embeddings.json", "r") as json_file:
            data = json.load(json_file)
    except:
        print("=================================== ERROR ===================================")
        print("                 JSONFile Contaning the Audio Embedding Not FOUND")
        print("-"*50)
        return label_names, embeddings

    
    # Extract the label names and embeddings for each speaker
    for entry in data:
        label_name = entry["label_name"]
        embedding = np.array(entry["audio_embeddings"]).reshape(1, 192)
    
        label_names.append(label_name)
        embeddings.append(embedding)

    return label_names, embeddings



def get_large_audio_chunks_on_silence(path):

    # video_clip = mp.VideoFileClip(path)

    # # Extract audio from the video in memory
    # audio_clip = video_clip.audio
    # audio_file_path="a.wav"
    # audio_clip.write_audiofile(audio_file_path)
    # video_clip.close()
    # # Close the audio clip
    # audio_clip.close()
    # open the audio file using pydub
    sound = AudioSegment.from_file(path)  

    # split audio sound where silence is 500 milliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len=1000,
        # adjust this per requirement
        silence_thresh=sound.dBFS - 14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    # print(chunks)
    # Initialize an empty list to hold start and end times of each chunk
    chunk_times = []

    # Initialize the start time for the first chunk
    start_time = 0

    # Loop through each chunk to calculate start and end times
    for chunk in chunks:
        # The end time of the current chunk is the start time plus the length of the chunk
        end_time = start_time + len(chunk)
        
        # Append the start and end times as a tuple to the chunk_times list
        chunk_times.append((start_time, end_time))
        
        # Update the start time for the next chunk
        start_time = end_time

    # Print the start and end times of each chunk
    # for i, (start, end) in enumerate(chunk_times):
    #     print(f"Chunk {i+1}: Start = {start} ms, End = {end} ms")
    return chunk_times
# get_large_audio_chunks_on_silence("TestData/ss.mp4")