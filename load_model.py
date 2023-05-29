import numpy as np
import os as os 
import cv2 as cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers

import torchaudio as torchaudio
import torch as torch
import librosa as librosa
from PIL import Image

import glob


if __name__ == "__main__":
    # Assemble the wav2vec model
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    wav2vec = bundle.get_model().to(device)

    # Assemble the model TODO
    with tf.device('/CPU:0'):
        model = keras.models.Sequential()
        model.add(layers.Conv2D(8,(5,5), strides=(1,1), padding="valid", activation='relu', input_shape=(64,448,3)))
        model.add(layers.MaxPool2D((2,2)))
        model.add(layers.Conv2D(16, 5, activation='relu'))
        model.add(layers.MaxPool2D((2,2)))
        model.add(layers.Conv2D(32,5, activation='relu'))
        model.add(layers.MaxPool2D((2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(100, activation='relu'))
        model.add(layers.Dense(2, activation='softmax'))

    model.load_weights('model.h5') # TODO name of model
    i = 0
    wav_files = glob.glob(os.path.join("input", '*.wav')) # TODO mp3? 
    for wav_file in wav_files:
        i += 1
        print("Processing:", wav_file, i)

        # Load the WAV file, cut the first 3 seconds and save
        waveform, sample_rate = torchaudio.load(wav_file) 
        num_samples = int(3 * sample_rate)
        waveform = waveform[:, :num_samples]
        audio_np = waveform.numpy()[0]
        audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=22050)
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        # torchaudio.save("temp/input_cut.wav", audio_tensor, 22050)
        torchaudio.save("input_cut/"+str(i)+"_cut.wav", audio_tensor, 22050)

        # Process the audio and convert it to an image
        with torch.inference_mode():
            emission, _ = wav2vec(audio_tensor.to(device))
            im = Image.fromarray(np.array(emission).astype('uint8'), mode='RGB')
            im.save("temp/"+str(i)+"_image.jpeg")

        # Load the image and prepare it for the model
        images = []
        file = "temp/"+str(i)+"_image.jpeg"
        temp_img = cv2.imread(file)
        temp_img = cv2.resize(temp_img, (448,64))
        images.append(temp_img)
        images = np.array(images)
        images = images.astype('float32')/255.0

        # Use the model to predict output values
        input_tensor = images
        output_values = model.predict(input_tensor)
        
        print("output values" , output_values)
        # use one line trinary operator to print male or female
        # TODO double check if the indexs are correct
        gender = "male" if output_values[0][0] > output_values[0][1] else "female"
        print(gender, str(i))
    # Remove all files in the temp folder
    for file in os.listdir("temp"):
        os.remove(os.path.join("temp", file))
