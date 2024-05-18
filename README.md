## Bioacoustic data compression

This repository is used to create an array spectrograms $X$ from any audio format (mp3, flac, ogg, aac) of animal vocalizations to train a CNN classifier for bioacoustic monitoring project.
The objective is to be able evaluate how well a classifier along with compressed data perform compared with the original data. 
The data is always recorded and saved as a waveform (.wav). You can use any tools to convert the audio but here we are created our own converter to facilitate the task of converting multiples files in one folder just in one click. This app is created with PyQt5.


  ![guiiapp](https://github.com/milanto-hery/audio_to_input_cnn/assets/78157308/7395bed5-d94d-4a66-8f9e-ac9d5d0a253f)
