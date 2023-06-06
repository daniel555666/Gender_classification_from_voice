import numpy as np
import os as os 
import cv2 as cv2

import torchaudio as torchaudio
import torch as torch
import librosa as librosa
from pydub import AudioSegment # to convert mp3

import glob
import json
import sys

from torch import nn
import torch

import matplotlib.pyplot as plt


# The model

NUM_OF_SPEAKERS = 2
DROP_OUT = 0.5

class Convolutional_Speaker_Identification(nn.Module):
    def cal_paddind_shape(self, new_shape, old_shape, kernel_size, stride_size):
        return (stride_size * (new_shape - 1) + kernel_size - old_shape) / 2

    def __init__(self, is_spec=False):
        super().__init__()
        if is_spec:
            self.conv_2d_1 = nn.Conv2d(3, 96, kernel_size=(7, 7), stride=(2, 2), padding=2) # spectrogram
        else:
            self.conv_2d_1 = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=2) # emission
        self.bn_1 = nn.BatchNorm2d(96)
        self.max_pool_2d_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv_2d_2 = nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.bn_2 = nn.BatchNorm2d(256)
        self.max_pool_2d_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv_2d_3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=2)
        self.bn_3 = nn.BatchNorm2d(384)

        self.conv_2d_4 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=2)
        self.bn_4 = nn.BatchNorm2d(256)

        self.conv_2d_5 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=2)
        self.bn_5 = nn.BatchNorm2d(256)
        self.max_pool_2d_3 = nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2))

        if is_spec:
            self.conv_2d_6 = nn.Conv2d(256, 4096, kernel_size=(9, 1), padding=2) # spectrogram
        else:
            self.conv_2d_6 = nn.Conv2d(256, 4096, kernel_size=(9, 1), padding=0) # emission
        self.drop_1 = nn.Dropout(p=DROP_OUT)

        self.global_avg_pooling_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.dense_1 = nn.Linear(4096, 1024)
        self.drop_2 = nn.Dropout(p=DROP_OUT)

        self.dense_2 = nn.Linear(1024, NUM_OF_SPEAKERS)

    def forward(self, X):
        x = nn.ReLU()(self.conv_2d_1(X))
        x = self.bn_1(x)
        x = self.max_pool_2d_1(x)

        x = nn.ReLU()(self.conv_2d_2(x))
        x = self.bn_2(x)
        x = self.max_pool_2d_2(x)

        x = nn.ReLU()(self.conv_2d_3(x))
        x = self.bn_3(x)

        x = nn.ReLU()(self.conv_2d_4(x))
        x = self.bn_4(x)

        x = nn.ReLU()(self.conv_2d_5(x))
        x = self.bn_5(x)
        x = self.max_pool_2d_3(x)

        x = nn.ReLU()(self.conv_2d_6(x))
        x = self.drop_1(x)
        x = self.global_avg_pooling_2d(x)

        x = x.view(-1, x.shape[1])  # output channel for flatten before entering the dense layer
        x = nn.ReLU()(self.dense_1(x))
        x = self.drop_2(x)

        x = self.dense_2(x)
        y = nn.LogSoftmax(dim=1)(x)   # consider using Log-Softmax

        return y

def record():
    import sounddevice as sd
    from scipy.io.wavfile import write
    import os

    # set the sampling rate and duration
    fs = 44100  # Sample rate 
    seconds = 3  # Duration of recording

    print("Recording...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording finished")

    # specify the output file name
    output_file = "input/output.wav"

    # Save as WAV file 
    write(output_file, fs, myrecording)

    print(f"Saved to {os.path.abspath(output_file)}") 

if __name__ == "__main__":
    # create directories
    if not os.path.exists("input_cut"):
        os.makedirs("input_cut")
    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists("input"):
        os.makedirs("input")

    if len(sys.argv) > 1:
        record()
    temp_path = "temp"

    # Assemble the wav2vec model
    torch.random.manual_seed(0)
    device = torch.device("cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    wav2vecModel = bundle.get_model().to(device)

    # model_path = "model_spanish3_spec.pt"
    model_path = "model_spec_all.pt"
    # model_path = "model_emission_all.pt"
    # model_path = "model_spec_arabic.pt"
    model = Convolutional_Speaker_Identification(is_spec=("spec" in model_path))

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    size = 29,449
    i = 0
    wav_files = glob.glob(os.path.join("input", '*')) 
    for wav_file in wav_files:
        i += 1
        print("Processing:", wav_file, i)
        if wav_file.endswith(".mp3"):
            sound = AudioSegment.from_mp3(wav_file)
            sound.export(temp_path+"/temp_input_file.wav", format="wav")
        elif wav_file.endswith(".wav"):
            sound = AudioSegment.from_wav(wav_file)
            sound.export(temp_path+"/temp_input_file.wav", format="wav")
        

        if "spec" in model_path:
            import pickle
            y, sr = librosa.load(wav_file)  # Load audio file
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)  # Compute spectrogram

            plt.clf()
            librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            plt.tight_layout()

            plt.savefig(temp_path+"/temp_input_file.png")
            plt.close()

            temp_img = cv2.imread(temp_path+"/temp_input_file.png")
            size = 320,240

            temp_img = cv2.resize(temp_img, size)
            temp_img=np.transpose(temp_img)

            images = []
            images.append(temp_img)
            images = np.array(images)
            images_tensor = torch.tensor(images, dtype=torch.float).to(device)  # Convert to tensor and move to the appropriate device

            output_values = model(images_tensor)



            # for i in images:
            #     with open(temp_path+"/temp_pkl_file.pkl", "wb") as f:
            #         pickle.dump((i,i), f)
            
            # with open(temp_path+"/temp_pkl_file.pkl", "rb") as f:
            #     images, _ = pickle.load(f)
            #     data_x = torch.tensor(images, dtype=torch.float32).to(device)  # Convert to tensor and move to the appropriate device
            #     _ = torch.tensor(_, dtype=torch.long).to(device)  # Convert to tensor and move to the appropriate device
            #     dataset = torch.utils.data.TensorDataset(data_x, _)
            #     test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
            
            # for images, _ in test_loader:
            #     images = images.to(torch.float)
                # data_x = data_x.unsqueeze(3)
                # data_x = data_x.to(torch.float)
            # images_tensor = torch.tensor(images[0], dtype=torch.float).to(device)  # Convert to tensor and move to the appropriate device
            # # images = images.to(torch.float) 
            # output_values = model(images_tensor)
            print(output_values)
        else:
            with torch.inference_mode():
                waveform, sample_rate = torchaudio.load(temp_path+"/temp_input_file.wav")
                waveform = waveform.to(device)
                emission, _ = wav2vecModel(waveform)

                images = []
                temp_np_array = np.array(emission)
                temp = [cv2.resize(temp_np_array[0], size)]
                temp_img = np.array(temp)
                images.append(temp_img)

                images_tensor = torch.tensor(images, dtype=torch.float32).to(device)  # Convert to tensor and move to the appropriate device
                output_values = model(images_tensor)
                print(output_values)


        gender = "male" if output_values[0][0] > output_values[0][1] else "female"
        print(gender, str(i))
    # Remove all files in the temp folder
    for file in os.listdir("temp"):
        os.remove(os.path.join("temp", file))
