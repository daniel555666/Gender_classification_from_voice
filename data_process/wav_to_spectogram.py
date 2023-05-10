#@title Default title text
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from pydub import AudioSegment # to convert mp3

import glob
import os


russian_path = "data/russian_data"
temp_path = "temp"
output_path = "data/json_russian_data"

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

i = 0

female_folder = glob.glob(os.path.join(russian_path, 'female_rusit'))[0]
wav_files = glob.glob(os.path.join(female_folder, '*.mp3'))
for wav_file in wav_files:
  i += 1
  # if i == 5: # REMOVE TO PROCESS ALL DATA
    # break
  print("proccessing..", wav_file, i)
  sound = AudioSegment.from_mp3(wav_file)
  sound.export(temp_path+"/temp.wav", format="wav")

  sample_rate, samples = wavfile.read(temp_path+"/temp.wav")
  frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

  plt.pcolormesh(times, frequencies, spectrogram)
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.savefig(output_path+"/W/"+str(i)+'.png')

male_folder = glob.glob(os.path.join(russian_path, 'male_rusit'))[0]
wav_files = glob.glob(os.path.join(male_folder, '*.mp3'))
for wav_file in wav_files:
  i += 1
  # if i == 10:  # REMOVE TO PROCESS ALL DATA
    # break
  print("proccessing..", wav_file, i)
  sound = AudioSegment.from_mp3(wav_file)
  sound.export(temp_path+"/temp.wav", format="wav")

  sample_rate, samples = wavfile.read(temp_path+"/temp.wav")
  frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

  plt.pcolormesh(times, frequencies, spectrogram)
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.savefig(output_path+"/M/"+str(i)+'.png')
