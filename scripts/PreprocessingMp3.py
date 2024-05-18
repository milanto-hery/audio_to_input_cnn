#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-

# Import Libraries
import os
import numpy as np
import librosa.display
import librosa
import pickle
import pandas as pd
import time
from scripts.AnnotationClassic import *

class PreprocessingMp3:
    
    def __init__(self, audio_path, annotations_path, training_files, segment_duration, 
                 positive_class, background_class,            
                 n_fft, hop_length, n_mels, f_min, f_max, file_type, audio_extension):
        self.segment_duration = segment_duration
        self.positive_class = positive_class
        self.background_class = background_class
        #self.species_folder = species_folder
        self.audio_path = audio_path
        self.annotations_path = annotations_path
        #self.saved_data_path = saved_data_path
        self.training_files = training_files
        self.n_ftt = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.file_type = file_type
        self.audio_extension = audio_extension
        
    #def update_audio_path(self, audio_path):
        #self.audio_path = self.audio_path
        
    def read_audio_file(self, file_name):
        '''
        file_name: string, name of file including extension, e.g. "audio1.wav"
        
        '''
        # Get the path to the file
        audio_folder = os.path.join(file_name)
        
        # Read the amplitudes and sample rate
        audio_amps, audio_sample_rate = librosa.load(audio_folder, sr=None)
        
        return audio_amps, audio_sample_rate
    

    def convert_single_to_image(self, audio):
        '''
        Convert amplitude values into a mel-spectrogram.
        '''
        S = librosa.feature.melspectrogram(audio, n_fft=self.n_ftt,hop_length=self.hop_length, 
                                           n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max)
        
        
        image = librosa.core.power_to_db(S)
        image_np = np.asmatrix(image)
        image_np_scaled_temp = (image_np - np.min(image_np))
        image_np_scaled = image_np_scaled_temp / np.max(image_np_scaled_temp)
        mean = image.flatten().mean()
        std = image.flatten().std()
        eps=1e-8
        spec_norm = (image - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
        S1 = spec_scaled
        
    
        # 3 different input
        return S1

    def convert_all_to_image(self, segments):
        '''
        Convert a number of segments into their corresponding spectrograms.
        '''
        spectrograms = []
        for segment in segments:
            spectrograms.append(self.convert_single_to_image(segment))

        return np.array(spectrograms)
    
    def add_extra_dim(self, spectrograms):
        '''
        Add an extra dimension to the data so that it matches
        the input requirement of Tensorflow.
        '''
        spectrograms = np.reshape(spectrograms, 
                                  (spectrograms.shape[0],
                                   spectrograms.shape[1],
                                   spectrograms.shape[2],1))
        return spectrograms
    
        
    def getXY(self, audio_amplitudes, sample_rate, start_sec, annotation_duration_seconds, label, verbose):
        '''
        Extract a number of segments based on the user-annotations.
        If possible, a number of segments are extracted provided
        that the duration of the annotation is long enough. The segments
        are extracted by shifting by 1 second in time to the right.
        Each segment is then augmented a number of times based on a pre-defined
        user value.
        '''

        if verbose == True:
            print ('start_sec', start_sec)
            print ('annotation_duration_seconds', annotation_duration_seconds)
            print ('self.segment_duration ', self.segment_duration )
            
        X_segments = []
        Y_labels = []
            
        # Calculate how many segments can be extracted based on the duration of
        # the annotated duration. If the annotated duration is too short then
        # simply extract one segment. If the annotated duration is long enough
        # then multiple segments can be extracted.
        if annotation_duration_seconds-self.segment_duration < 0:
            segments_to_extract = 1
        else:
            segments_to_extract = annotation_duration_seconds-self.segment_duration+1
            
        if verbose:
            print ("segments_to_extract", segments_to_extract)
            
        if label in self.background_class:
            if segments_to_extract > 10:
                segments_to_extract = 10

        for i in range (0, segments_to_extract):
            if verbose:
                print ('Semgnet {} of {}'.format(i, segments_to_extract-1))
                print ('*******************')
            # The correct start is with respect to the location in time
            # in the audio file start+i*sample_rate
            start_data_observation = start_sec*sample_rate+i*(sample_rate)
            # The end location is based off the start
            end_data_observation = start_data_observation + (sample_rate*self.segment_duration)
            
            # This case occurs when something is annotated towards the end of a file
            # and can result in a segment which is too short.
            if end_data_observation > len(audio_amplitudes):
                continue

            # Extract the segment of audio
            X_audio = audio_amplitudes[start_data_observation:end_data_observation]

            # Determine the actual time for the event
            start_time_seconds = start_sec + i

            if verbose == True:
                print ('start frame', start_data_observation)
                print ('end frame', end_data_observation)
            
            # Extend the augmented segments and labels (and the metadata)
            X_segments.append(X_audio)
            Y_labels.append(label)

        return X_segments, Y_labels
        
    def create_dataset_mp3(self, saved_segments, verbose):
        '''
        Create X and Y values which are inputs to a ML algorithm.
        Annotated files (.svl) are read and the corresponding audio file (.wav)
        is read. A low pass filter is applied, followed by downsampling. A 
        number of segments are extracted and augmented to create the final dataset.
        Annotated files (.svl) are created using SonicVisualiser and it is assumed
        that the "boxes area" layer was used to annotate the audio files.
        '''
        
        # Keep track of how many calls were found in the annotation files
        total_calls = 0

        # Initialise lists to store the X and Y values
        
        X_wav = []
        X_S =[]
        X_calls = []
        Y_calls = []
        
        if verbose == True:
            print ('Annotations path:',self.annotations_path+"*.svl")
            print ('Audio path',self.audio_path+"*.mp3")
        
        # Read all names of the training files
        training_files = pd.read_csv(self.training_files, header=None)
        
        # Iterate over each annotation file
        for training_file in training_files.values:
            
            file = training_file[0]
            
            if self.file_type == 'svl':
                # Get the file name without paths and extensions
                file_name_no_extension = file
                print ('file_name_no_extension', file_name_no_extension)
            if self.file_type == 'raven_caovitgibbons':
                file_name_no_extension = file[file.rfind('-')+1:file.find('.')]
                
            print ('Processing:',file_name_no_extension)
            
            reader =AnnotationClassic(file, self.annotations_path, self.audio_path, self.file_type, self.audio_extension)
            #(self, annotation_file_name, path_annotations, path_audio, file_type, audio_extension)

            # Check if the .wav file exists before processing
            print(self.audio_path+file_name_no_extension+self.audio_extension)
            if os.path.exists(self.audio_path+file_name_no_extension+self.audio_extension): 
                print('Found file')
                
                # Read audio file
                audio_amps, original_sample_rate = self.read_audio_file(self.audio_path+file_name_no_extension+self.audio_extension)
                print('Original sampling rate: ', original_sample_rate)

                df, audio_file_name = reader.get_annotation_information()

                print('Reading annotations...')
                for _, row in df.iterrows():

                    start_seconds = int(round(row['Start']))
                    end_seconds = int(round(row['End']))
                    label = row['Label']
                    annotation_duration_seconds = end_seconds - start_seconds
                    print("Extract audio segments...")
                    # Extract augmented audio segments and corresponding binary labels
                    X_data, y_data = self.getXY(audio_amps, original_sample_rate, start_seconds, annotation_duration_seconds, label, verbose)
                    print("Convert to spectrograms...")
                    # Convert audio amplitudes to spectrograms
                    t_s = time.time()
                    X_data_S = self.convert_all_to_image(X_data)
                    print('Shape:',X_data_S.shape )
                    t_e = time.time()
                    t_process = t_e-t_s 
                    print("done.\n")
                    batch_times = []  # To store batch processing times.
                    spec_times = []
                
                    X_wav.extend(X_data) #audio
                    X_S.extend(X_data_S) # full raw spectrograms
                    #X_calls.extend(X_data_S) # narrow band
                    #t_spec.append(t_process)
                    Y_calls.extend(y_data)

                    print('-----------------------------------successfully--------------------------------')


                # Average batch processing time.
                print(f"Time to convert spectrograms: {t_process:.2f} seconds")

                #print(f'Processing time for {file_name_no_extension}: {t_process:.2f} seconds')
                    
        # Convert to numpy arrays
        X_audio, X_Full, Y_calls = np.array(X_wav), np.array(X_S), np.array(Y_calls)
        print('X_full shape', X_Full.shape)
        
        # Create a folder to save the data
        #---------------------------------------------------------------------------------------------------------------------------       
        if not os.path.exists(saved_segments):
            os.makedirs(saved_segments)


        # save all the data in pickle file
        pickle_file = os.path.join(saved_segments, f"X_wav.pkl")
        with open(pickle_file, 'wb') as file:
            pickle.dump(X_audio, file)
            
        pickle_file = os.path.join(saved_segments, f"X_raw.pkl")
        with open(pickle_file, 'wb') as file:
            pickle.dump(X_Full, file)

        pickle_file = os.path.join(saved_segments, f"Y_S.pkl")
        with open(pickle_file, 'wb') as file:
            pickle.dump(Y_calls, file)

        # ------------------------------------------------------------------------------------------------
        return X_Full, Y_calls


