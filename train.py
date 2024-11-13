import torch
from scipy.spatial.distance import cdist
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

from utils import *


traning_audios_folder = "audios" 
'''path of folder that contains the audio samples for traning like |-Audio_Folder
                                                                        |-John Paul
                                                                            |audio.wav
                                                                            |voice.wav
                                                                        |-Alice Nik
                                                                            |audio.wav
                                                                            |recod.wav                                                                    
'''
# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available. Using GPU.")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")
model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device(device))
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

def training(label , audio_file):
    audio = Audio(sample_rate=16000, mono="downmix")
    speaker_voice = Segment(0., float(audio.get_duration(audio_file)))
    waveform, sample_rate = audio.crop(audio_file, speaker_voice)
    embedding = model(waveform[None])
    audio_path = save_audio_file(label, audio_file)
    speaker = [
        { "label_name": label, "embeddings": embedding, "audio_file": audio_path}
    ]
    save_embeddings(speaker)



def main():
    '''Traning'''
    root_folder = traning_audios_folder
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
            try:
                print(audio_path)
                training(person, audio_path)
            except Exception as e:
                print(e)
        print("==="*20)
    print("Emebedding are extracted and Saved to Json File")

if __name__ == "__main__":
    main()