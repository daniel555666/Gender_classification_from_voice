import torch
import torchaudio
from torchaudio.utils import download_asset
from PIL import Image
import numpy as np

import json

from pydub import AudioSegment # to convert mp3

import glob
import os
import sys
if __name__ == "__main__":
   arg1 = sys.argv[1]
   if arg1 == "all":
      for language in ['arabic', 'english', 'french', 'spanish']:
      # for lanaugage in ['russian', 'arabic', 'english', 'french', 'spanish']:
         print("processing language:", language)
         sys.argv[1] = language
         os.system("python3 data_process/emmissions_to_json.py "+language)
      exit(0)
   all_languages = ['russian', 'arabic', 'english', 'french', 'spanish']
   if arg1 not in all_languages:
      print("bad language")
      exit(1)
   language_path = "data/"+arg1+"_data"
   temp_path = "temp"
   output_path = "data/json_"+arg1+"_data"
   if not os.path.exists(output_path):
      os.makedirs(output_path)
   if not os.path.exists(output_path+"/W"):
      os.makedirs(output_path+"/W")
   if not os.path.exists(output_path+"/M"):
      os.makedirs(output_path+"/M")

   torch.random.manual_seed(0)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print("device", device)
   bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
   model = bundle.get_model().to(device)

   i = 0


   female_folder = glob.glob(os.path.join(language_path, "female_"+arg1))[0]

   print(female_folder)
   wav_files = glob.glob(os.path.join(female_folder, '*.mp3'))
   for wav_file in wav_files:
      i += 1
      # if i == 5: # REMOVE TO PROCESS ALL DATA
      #   break
      print("proccessing..", wav_file, i)
      sound = AudioSegment.from_mp3(wav_file)
      sound.export(temp_path+"/"+arg1+"_temp_emmision_to_json.wav", format="wav")
      
      waveform, sample_rate = torchaudio.load(temp_path+"/"+arg1+"_temp_emmision_to_json.wav")
      waveform = waveform.to(device)

      with torch.inference_mode():
         emission, _ = model(waveform)
         out_file = open(output_path+"/W/"+str(i)+'.json', "w")
         json.dump(emission.tolist(), out_file)

   male_folder = glob.glob(os.path.join(language_path, "male_"+arg1))[0]
   wav_files = glob.glob(os.path.join(male_folder, '*.mp3'))
   for wav_file in wav_files:
      i += 1
      # if i == 10:  # REMOVE TO PROCESS ALL DATA
      #   break
      print("proccessing..", wav_file, i)
      sound = AudioSegment.from_mp3(wav_file)
      sound.export(temp_path+"/"+arg1+"_temp_emmision_to_json.wav", format="wav")

      waveform, sample_rate = torchaudio.load(temp_path+"/"+arg1+"_temp_emmision_to_json.wav")
      waveform = waveform.to(device)

      with torch.inference_mode():
         emission, _ = model(waveform)
         out_file = open(output_path+"/M/"+str(i)+'.json', "w")
         json.dump(emission.tolist(), out_file)