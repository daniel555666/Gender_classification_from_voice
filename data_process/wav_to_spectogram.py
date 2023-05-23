import os
import sys
import librosa
import matplotlib.pyplot as plt
import numpy as np
import tkinter
import tkinter as tk

if __name__ == "__main__":
  arg1 = sys.argv[1]
  all_languages = ['russian', 'arabic', 'english', 'french', 'spanish']
  if arg1 == "all":
    for language in all_languages:
      print("processing language:", language)
      sys.argv[1] = language
      os.system("python3 data_process/wav_to_spectogram.py "+language)
    exit(0)
  if arg1 not in all_languages:
    print("bad language")
    exit(1)
  language_path = "data/"+arg1+"_data"
  temp_path = "temp"
  output_path = "data/spectogram_"+arg1+"_data"
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


if __name__ == "__main__":
    arg1 = sys.argv[1]
    all_languages = ['russian', 'arabic', 'english', 'french', 'spanish']
    if arg1 == "all":
        # Process all languages
        for language in all_languages:
            print("Processing language:", language)
            sys.argv[1] = language
            os.system("python3 data_process/wav_to_spectrogram.py " + language)
        exit(0)

    if arg1 not in all_languages:
        print("Bad language")
        exit(1)

    language_path = "data/" + arg1 + "_data"
    output_path = "data/spectrogram_" + arg1 + "_data"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(output_path + "/W"):
        os.makedirs(output_path + "/W")
    if not os.path.exists(output_path + "/M"):
        os.makedirs(output_path + "/M")

    i = 0
    print("before female")
    female_folder = os.path.join(language_path, "female_" + arg1)
    wav_files = [file for file in os.listdir(female_folder) if file.endswith('.mp3')]
    for wav_file in wav_files:
        i += 1
        # if i == 5:  # REMOVE TO PROCESS ALL DATA
        #     break
        print("Processing:", wav_file, i)
        audio_path = os.path.join(female_folder, wav_file)
        y, sr = librosa.load(audio_path)  # Load audio file
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)  # Compute spectrogram

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()

        plt.savefig(output_path + "/W/" + str(i) + '.png')
        plt.close()

    # Repeat the process for the male folder

    male_folder = os.path.join(language_path, "male_" + arg1)
    wav_files = [file for file in os.listdir(male_folder) if file.endswith('.mp3')]
    for wav_file in wav_files:
        i += 1
        # if i == 10:  # REMOVE TO PROCESS ALL DATA
        #     break
        print("Processing:", wav_file, i)
        audio_path = os.path.join(male_folder, wav_file)
        y, sr = librosa.load(audio_path)  # Load audio file
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)  # Compute spectrogram

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()

        plt.savefig(output_path + "/M/" + str(i) + '.png')
        plt.close
