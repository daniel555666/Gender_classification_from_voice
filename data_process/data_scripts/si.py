import os
from mutagen.mp3 import MP3

directories = ['female_spanish', 'male_spanish']

duration_count = {}


for directory in directories:
    for filename in os.listdir(directory):
        if filename.endswith(".mp3"):
            filepath = os.path.join(directory, filename)
            audio = MP3(filepath)
            duration = int(audio.info.length)

            if duration in duration_count:
                duration_count[duration] += 1
            else:
                duration_count[duration] = 1


# Print the duration and count for each duration
for duration, count in duration_count.items():
    print(f"Duration: {duration} seconds, Count: {count}")