# -*- coding: utf-8 -*-

import json

from scripts.PreprocessingAac import *
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
    audio_extension = species_params['audio_aac']
    n_fft = species_params['n_fft']
    hop_length = species_params['hop_length']
    n_mels = species_params['n_mels']
    f_min = species_params['f_min']
    f_max = species_params['f_max']
    annotations_path = './Annotations/'
    training_files = './DataFiles/TrainingFiles.txt'

    # List of bitrates to process
    bitrates = [8, 16, 32, 64, 128]
    # Base directory contains all audio folders of each bitate and the folders to store all the the data
    in_dir_aac = './In_Data/WAV_AAC/'
    out_dir_aac = './Out_Data/AAC/'

    for l in bitrates:
        input_aac = os.path.join(in_dir_aac, f'AAC_{l}')
        output_aac = os.path.join(out_dir_aac, f'Saved_{l}')
        
        if os.path.exists(input_aac):
            os.makedirs(output_aac, exist_ok=True) 
            process_aac = PreprocessingAac(input_aac+"/", annotations_path, training_files, segment_duration,
                    positive_class, negative_class, n_fft, hop_length, n_mels, f_min, f_max, file_type, audio_extension)
            process_aac.create_dataset_aac(output_aac, False)
        else:
            print(f"Folder {input_aac} not found. Skipping...")


if __name__ == "__main__":
    main()



