import numpy as np
import os as os 
import cv2 as cv2

import torchaudio as torchaudio
import torch as torch
import librosa as librosa
from pydub import AudioSegment # to convert mp3

import glob
import json

from torch import nn

# The model

NUM_OF_SPEAKERS = 2
DROP_OUT = 0.5

class Convolutional_Speaker_Identification(nn.Module):
    def cal_paddind_shape(self, new_shape, old_shape, kernel_size, stride_size):
        return (stride_size * (new_shape - 1) + kernel_size - old_shape) / 2

    def __init__(self):
        super().__init__()
        self.conv_2d_1 = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=2)
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

        self.conv_2d_6 = nn.Conv2d(256, 4096, kernel_size=(9, 1), padding=0)
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


if __name__ == "__main__":
    # create directories
    if not os.path.exists("input_cut"):
        os.makedirs("input_cut")
    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists("input"):
        os.makedirs("input")
    temp_path = "temp"

    # Assemble the wav2vec model
    torch.random.manual_seed(0)
    device = torch.device("cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    wav2vecModel = bundle.get_model().to(device)

    model = Convolutional_Speaker_Identification()
    model_path ="model.all.pt"  
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    i = 0
    wav_files = glob.glob(os.path.join("input", '*.mp3')) 
    for wav_file in wav_files:
        i += 1
        print("Processing:", wav_file, i)
        sound = AudioSegment.from_mp3(wav_file)
        sound.export(temp_path+"/temp_input_file.wav", format="wav")
        
        waveform, sample_rate = torchaudio.load(temp_path+"/temp_input_file.wav")
        waveform = waveform.to(device)

        with torch.inference_mode():
            emission, _ = wav2vecModel(waveform)
            # features, _ = model.extract_features(waveform)
            # out_file = open(temp_path+"/temp_input_json_file.json", "w")
            # json.dump(emission.tolist(), out_file)
            # json.dump([element.tolist() for element in features], out_file)

            output = model(emission)
            print(output)
            gender = "male" if output_values[0][0] > output_values[0][1] else "female"
            print(gender, str(i))


        # # Use the model to predict output values
        # input_tensor = images
        # output_values = model.predict(input_tensor)
        
        # print("output values" , output_values)
        # # use one line trinary operator to print male or female
        # # TODO double check if the indexs are correct
        gender = "male" if output_values[0][0] > output_values[0][1] else "female"
        print(gender, str(i))
    # Remove all files in the temp folder
    for file in os.listdir("temp"):
        os.remove(os.path.join("temp", file))
