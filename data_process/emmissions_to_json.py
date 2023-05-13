import torch
import torchaudio
from torchaudio.utils import download_asset
from PIL import Image
import numpy as np

import json

from pydub import AudioSegment # to convert mp3

import glob
import os

russian_path = "data/russian_data"
temp_path = "temp"
output_path = "data/json_russian_data"

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

i = 0


female_folder = glob.glob(os.path.join(russian_path, 'female_russian'))[0]

print(female_folder)
wav_files = glob.glob(os.path.join(female_folder, '*.mp3'))
for wav_file in wav_files:
  i += 1
  # if i == 5: # REMOVE TO PROCESS ALL DATA
    # break
  print("proccessing..", wav_file, i)
  sound = AudioSegment.from_mp3(wav_file)
  sound.export(temp_path+"/temp.wav", format="wav")
  
  waveform, sample_rate = torchaudio.load(temp_path+"/temp.wav")
  waveform = waveform.to(device)

  with torch.inference_mode():
      emission, _ = model(waveform)
      out_file = open(output_path+"/M/"+str(i)+'.json', "w")
      json.dump(emission.tolist(), out_file)

male_folder = glob.glob(os.path.join(russian_path, 'male_russian'))[0]
wav_files = glob.glob(os.path.join(male_folder, '*.mp3'))
for wav_file in wav_files:
  i += 1
  # if i == 10:  # REMOVE TO PROCESS ALL DATA
    # break
  print("proccessing..", wav_file, i)
  sound = AudioSegment.from_mp3(wav_file)
  sound.export(temp_path+"/temp.wav", format="wav")

  waveform, sample_rate = torchaudio.load(temp_path+"/temp.wav")
  waveform = waveform.to(device)

  with torch.inference_mode():
      emission, _ = model(waveform)
      out_file = open(output_path+"/M/"+str(i)+'.json', "w")
      json.dump(emission.tolist(), out_file)