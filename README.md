## Bioacoustic data compression

This repository is used to create a spectrograms array $X$ and and its labels $Y$ of any audio format (mp3, flac, ogg, aac) to train a CNN classifier for bioacoustic data.
The objective is to be able to evaluate how well any classifiers along the compressed data using all available data compression techniques perform compared with the original data. 
The data is always savedd as a waveform and all the conversion are achieved by the GUI app created with PyQt.

