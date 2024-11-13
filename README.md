# speaker-detection-api
Inference API for Speaker detection using speechbrain

This repository contains a Speaker Detection model capable of identifying specific speakers in audio recordings. It utilizes deep learning techniques to extract features and make predictions based on voice samples.

Installation
1- Clone the repository:                             git clone https://github.com/AI-TEAM-R-D-Models/speaker-detection-api.git
2- Navigate into the cloned directory:               cd SpeakerDetection
3- Install the required dependencies:                pip install -r req.txt



Training

To train the model, follow these steps:

1- Organize your training data:

            * Place audio recordings of each speaker in separate folders within the "training" directory.
            * Each folder should be named after the respective speaker.
        Example structure:
                        |- training
                                |- John Paul
                                    |  |- audio1.wav
                                    |  |- audio2.wav
                                    |  |- ...
                                |- Alice Nik
                                    |- audio1.wav
                                    |- audio2.wav
                                    |- ...

2- Start the training process:

                 python train.py


                This command initiates the training process, and the trained model data will be stored in the "audios" folder. Embeddings will also be saved in the "Speaker_embeddings.json" file.

3- Training Complete.



Prediction

To use the trained model for prediction, follow these steps:

1- Prepare your validation data:

                    Place the audio recordings for prediction in the "validation" folder.

2- Execute the prediction script:

                 python predict.py

        This command will generate prediction results based on the provided audio files.
