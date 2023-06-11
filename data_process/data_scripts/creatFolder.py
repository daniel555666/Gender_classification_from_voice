import os
import shutil

# read the "audio" file and extract the names of the MP3 files listed in it
with open("name_audio", "r") as f:
    mp3_names = [line.strip() for line in f]

# create a new folder to store the selected MP3 files
# if not os.path.exists("selected_mp3"):
    os.mkdir("female_spanish")

# iterate over the MP3 files in the original package
for file_name in os.listdir("clips"):
    # check if the MP3 file name matches one of the names extracted from the "audio" file
    if file_name in mp3_names:
        # move the MP3 file to the new package
        shutil.move(os.path.join("clips", file_name), os.path.join("female_spanish", file_name))