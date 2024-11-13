import torch
from scipy.spatial.distance import cdist
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from IPython.display import Audio as player
from utils import *

confidence = 0.8
validation_data_folder = "validation"
'''path of folder that contains the audio samples for validation like |-Validation_Audio_Folder
                                                                        |-John Paul
                                                                            |audio.wav
                                                                            |voice.wav
                                                                        |-Alice Nik
                                                                            |audio.wav
                                                                            |recod.wav                                                                    
'''
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

def main():
    root_folder = validation_data_folder
    audio_dict = {}
    for person_folder in os.listdir(root_folder):
        person_folder_path = os.path.join(root_folder, person_folder)
        if os.path.isdir(person_folder_path):
            audio_paths = []
            for audio_file in os.listdir(person_folder_path):
                audio_file_path = os.path.join(person_folder_path, audio_file)
                if os.path.isfile(audio_file_path):
                    audio_paths.append(audio_file_path)
            if audio_paths:
                audio_dict[person_folder] = audio_paths
   
    for person, audio_paths in audio_dict.items():
        print(f"{person}:")
        for audio_path in audio_paths:
            person, score = recognition(audio_path)
            print(f"This is voice of {person} with the similarity of {(1-score)*100}%")
        print("***"*20)

if __name__ == "__main__":
    main()