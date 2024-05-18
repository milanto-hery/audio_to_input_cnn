# -*- coding: utf-8 -*-

import json
from scripts.PreprocessingFlac import *
import time
import os

def load_json_config(file_name):
    try:
        with open(file_name, 'r') as file:
            config = json.load(file)
        return config
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file '{file_name}': {e}")
        return None

# Set the default working directory
my_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(my_dir)

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
    audio_extension = species_params['audio_flac']
    n_fft = species_params['n_fft']
    hop_length = species_params['hop_length']
    n_mels = species_params['n_mels']
    f_min = species_params['f_min']
    f_max = species_params['f_max']
    annotations_path = './Annotations/'
    training_files = './DataFiles/TrainingFiles.txt'
    levels = [0, 6, 8, 10] #list of quality levels to process

    # Base directory contains all audio folders of each bitate and the folders to store all the the data
    in_dir_flac = './In_Data/WAV_FLAC/'
    out_dir_flac = './Out_Data/FLAC/'

    for j in levels:
        input_flac = os.path.join(in_dir_flac, f'FLAC_{j}')
        output_flac = os.path.join(out_dir_flac, f'Saved_{j}')
        
        if os.path.exists(input_flac):
            os.makedirs(output_flac, exist_ok=True) 
            process_flac = PreprocessingFlac(input_flac+"/", annotations_path, training_files, segment_duration,
                    positive_class, negative_class, n_fft, hop_length, n_mels, f_min, f_max, file_type, audio_extension)
            X_flac, Y = process_flac.create_dataset_flac(output_flac, False)
            time.sleep(3)
        else:
            print(f"Folder {input_flac} not found. Skipping...")

if __name__ == "__main__":
    main()



