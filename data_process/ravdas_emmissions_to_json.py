import torch
import torchaudio
from torchaudio.utils import download_asset
from PIL import Image
import numpy as np

import os
import glob

import json

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

parent_folder = 'data/'
actor_folders = glob.glob(os.path.join(parent_folder, 'Actor_*'))
i = 0
for actor_folder in actor_folders:
    wav_files = glob.glob(os.path.join(actor_folder, '*.wav'))
    destFolder = "W" if int(actor_folder[-2:])%2 == 0 else "M"
    os.makedirs("imgs/"+actor_folder, exist_ok=True)
    print("saving to "+"gender_data/"+destFolder+"/")
    for wav_file in wav_files:
        i += 1
        print("Processing:", wav_file)

        waveform, sample_rate = torchaudio.load(wav_file)
        waveform = waveform.to(device)

        with torch.inference_mode():
            emission, _ = model(waveform)
            out_file = open("json_gender_data/"+destFolder+"/"+str(i)+".json", "w")
            json.dump(emission.tolist(), out_file)
            print(emission)
            # im = Image.fromarray(np.array(emission).astype('uint8'), mode='RGB')
            # im.save("gender_data/"+destFolder+"/"+str(i)+".jpeg")

