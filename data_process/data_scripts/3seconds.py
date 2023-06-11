# import os
# import numpy as np
# import librosa
# import librosa.display
# import soundfile as sf
#
# audio_dir = r'male_spanish'
# out_dir = r'spanish_male'
# os.makedirs(out_dir, exist_ok=True)
#
# segment_dur_secs = 3
#
# for file_name in os.listdir(audio_dir):
#     if file_name.endswith('.mp3') or file_name.endswith('.wav'):
#         # Load the audio file
#         audio_file = os.path.join(audio_dir, file_name)
#         wave, sr = librosa.load(audio_file, sr=None)
#
#         # Split the audio into segments
#         segment_length = sr * segment_dur_secs
#         split = []
#         num_segments = int(np.ceil(segment_dur_secs))
#         t = wave[:segment_length]
#         split.append(t)
#
#         # Write each segment to a separate WAV file
#         recording_name = os.path.basename(file_name)[:-4]  # remove the file extension
#         for i, segment in enumerate(split):
#             out_file = f"{recording_name}.mp3"
#             sf.write(os.path.join(out_dir, out_file), segment, sr)
import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf

audio_dir = r'female_spanish'
out_dir = r'spanish_female'
os.makedirs(out_dir, exist_ok=True)

segment_dur_secs = 3
max_records = 24001
record_count = 0

for file_name in os.listdir(audio_dir):
    if file_name.endswith('.mp3') or file_name.endswith('.wav'):
        # Load the audio file
        audio_file = os.path.join(audio_dir, file_name)
        wave, sr = librosa.load(audio_file, sr=None)

        # Split the audio into segments
        segment_length = sr * segment_dur_secs
        split = []
        for s in range(0, len(wave), segment_length):
            t = wave[s: s + segment_length]
            if len(t) == segment_length:
                split.append(t)
                record_count += 1
                if record_count >= max_records:
                    break
        if record_count >= max_records:
            break

        # Write each segment to a separate WAV file
        recording_name = os.path.basename(file_name)[:-4]  # remove the file extension
        for i, segment in enumerate(split):
            out_file = f"{recording_name}_{i}.mp3"
            sf.write(os.path.join(out_dir, out_file), segment, sr)