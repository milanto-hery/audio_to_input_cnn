# -*- coding: utf-8 -*-

import json
from scripts.PreprocessingMp3 import *
import os

def load_json_config(file_name):
    try:
        with open(file_name, 'r') as file:
            config = json.load(file)
        return config
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file '{file_name}': {e}")
        return None
        
def main():
# Load configuration from JSON
    config = load_json_config('settings.json')
    if config:
        print(config)

    species_name = 'hainan_gibbon' # just specify the name of your species here, the parameters already define in the json
    species_params = config[species_name]

    segment_duration = species_params['segment_duration']
    positive_class = species_params['positive_class']
    negative_class = species_params['negative_class']
    file_type = species_params['file_type']
    audio_extension = species_params['audio_mp3']
    n_fft = species_params['n_fft']
    hop_length = species_params['hop_length']
    n_mels = species_params['n_mels']
    f_min = species_params['f_min']
    f_max = species_params['f_max']
    annotations_path = './Annotations/'
    training_files = './DataFiles/TrainingFiles.txt'
    bitrates = [8,16,32,64,128]

    # Base directory contains all audio folders of each bitate and the folders to store all the the data
    in_dir = './In_Data/WAV_MP3'
    out_dir = './Out_Data/MP3'

    for i in bitrates:
        input_mp3 = os.path.join(in_dir, f'FLAC_{i}')
        print("input audio:", input_mp3)
        output_mp3 = os.path.join(out_dir, f'Saved_{i}')
        
        if os.path.exists(input_mp3):
            os.makedirs(output_mp3, exist_ok=True) 
            process_mp3 = PreprocessingMp3(input_mp3+"/", annotations_path, training_files, segment_duration,
                    positive_class, negative_class, n_fft, hop_length, n_mels, f_min, f_max, file_type, audio_extension)
            X_mp3, Y = process_mp3.create_dataset_mp3(output_mp3, False)
            time.sleep(10)
        else:
            print(f"Folder {input_mp3} not found. Skipping...")

if __name__ == "__main__":
    main()
