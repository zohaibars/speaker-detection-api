import torch
from scipy.spatial.distance import cdist
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

from utils import *

confidence = 0.6


model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"))
audio = Audio(sample_rate=16000, mono="downmix")


def embeddings_distance(embedding1, embedding2):
    return cdist(embedding1, embedding2, metric="cosine")[0,0]
    
def get_part_embedding(audio_file, start_time, end_time, model=model, audio=audio):
    speaker_voice = Segment(float(start_time), float(end_time))
    waveform, sample_rate = audio.crop(audio_file, speaker_voice)
    embedding = model(waveform[None])
    return embedding

def audio_embedding(audio_file, model=model, audio=audio):
    speaker_voice = Segment(0., float(audio.get_duration(audio_file)))
    waveform, sample_rate = audio.crop(audio_file, speaker_voice)
    embedding = model(waveform[None])
    return embedding

def recognition(audio_path, verbose=False):
    embeddings = audio_embedding(audio_path)
    label_names, label_embeddings = load_embeddings()
    prediction = "UN_KNOW"
    score = 1 - confidence
    for label_name, label_embedding in zip(label_names, label_embeddings):
        if verbose:
            print("Person Name:", label_name)
        distance = embeddings_distance(embeddings, label_embedding)
        if verbose:
            print("Distance:", distance)
        if distance < 1 - confidence and distance < score:
            prediction = label_name
            score = distance
    return prediction, score


def recognition_audio_part(audio_path, verbose=False, start=0, end=0):
    embeddings = get_part_embedding(audio_file=audio_path,start_time=start, end_time=end)
    label_names, label_embeddings = load_embeddings()
    prediction = "UN_KNOW"
    score = 1 - confidence
    
    for label_name, label_embedding in zip(label_names, label_embeddings):
        if verbose:
            print("Person Name:", label_name)
        distance = embeddings_distance(embeddings, label_embedding)
        if verbose:
            print("Distance:", distance)
        if distance < 1 - confidence and distance < score:
            prediction = label_name
            score = distance
    return prediction, score

def main(audio_path):
    person, score = recognition(audio_path)
    print(f"This is voice of {person} with the similarity of {(1-score)*100}%")

def main_part(audio_path):
    chunk_times=get_large_audio_chunks_on_silence(audio_path)
    results = []
    for i, (start, end) in enumerate(chunk_times):
        start_sec = start/ 1000.0
        end_sec = end/ 1000.0
        # print(f"Chunk {i+1}: Start = {start_sec} s, End = {end_sec} s")
        person, score = recognition_audio_part(audio_path,start=start_sec,end=end_sec)
        # print(f"This is voice of {person} with the similarity of {(1-score)*100}%")
        results.append({
            "chunk": i + 1,
            "start": start_sec,
            "end": end_sec,
            "person": person,
            "similarity": (1 - score) * 100
        })
    
    # Print results as JSON
    # print(json.dumps(results, indent=4))
    return(json.dumps(results, indent=4))

# if __name__ == "__main__":
#     # main(audio_path="/home/waqar/MWaqar/SpeakerDetection/audios/Imran Khan/1.mp3")
#     main_part(audio_path="a.wav")