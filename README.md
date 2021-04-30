# curr_bio
Notebooks used for neural networks in Current Biology publication: Neurally driven synthesis of learned, complex vocalizations

All preprocessing functions are stored in spec_processing.py

Each notebook correspond to one network model:
1. predict_real_spec_mel_2layers_vanilla is the main LSTM model used to predict mel spectrogram slices from HVC activities.
3. predict_real_spec_mel_ffnn is the FFNN model used for comparison.
5. predict_real_spec_mel_2layers_warped is the same LSTM model but operated on data after dynamic warping, with or without a shuffling mask.

