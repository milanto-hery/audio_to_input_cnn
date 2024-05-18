# -*- coding: utf-8 -*-

import json
from scripts.PreprocessingOgg import *
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
    species_params = config[species_name]  # this for thyolo alethe

    segment_duration = species_params['segment_duration']
    positive_class = species_params['positive_class']
    negative_class = species_params['negative_class']
    file_type = species_params['file_type']
    audio_extension = species_params['audio_ogg']
    n_fft = species_params['n_fft']
    hop_length = species_params['hop_length']
    n_mels = species_params['n_mels']
    f_min = species_params['f_min']
    f_max = species_params['f_max']
    annotations_path = './Annotations/'
    training_files = './DataFiles/TrainingFiles.txt'
    levels = [0, 2, 6, 8] # List of compression levels to process

    # Base directory contains all audio folders of each bitrate and the folders to store all the data
    in_dir_ogg = './In_Data/WAV_OGG/'
    out_dir_ogg = './Out_Data/OGG/'

    for k in levels:
        input_ogg = os.path.join(in_dir_ogg, f'OGG_{k}')
        output_ogg = os.path.join(out_dir_ogg, f'Saved_{k}')
        
        if os.path.exists(input_ogg):
            os.makedirs(output_ogg, exist_ok=True)
            process_ogg = PreprocessingOgg(input_ogg+"/", annotations_path, training_files, segment_duration,
                    positive_class, negative_class, n_fft, hop_length, n_mels, f_min, f_max, file_type, audio_extension)
            process_ogg.create_dataset_ogg(output_ogg, False)
        else:
            print(f"Folder {input_ogg} not found. Skipping...")

if __name__ == "__main__":
    main()



