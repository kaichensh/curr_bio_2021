
# coding: utf-8

# In[2]:


get_ipython().magic(u'matplotlib inline')
import IPython.display
from ipywidgets import interact, interactive, fixed
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import scipy.ndimage
import socket
import os
import sys
import pandas as pd
import scipy.io as sio
import matplotlib
import scipy.signal as sg
import math
import scipy as sp
import pylab
import os
import wave
import scipy.io.wavfile
import h5py
import pickle
import tensorflow as tf
import random
import h5py
import collections
import bisect
import platform
import random
import datetime


# In[3]:


def locate_folders(bird_id, model_name):
#locate source folder
    hostname = socket.gethostname()
    
    if hostname == 'txori':
        data_folder = os.path.abspath('/mnt/cube/kai/data/hvc')
        results_folder = os.path.abspath('/mnt/cube/kai/results/'+model_name)
        repos_folder = os.path.abspath('/mnt/cube/kai/repositories')
        data_folder_zeke = os.path.abspath('/mnt/cube/earneodo/bci_zf/ss_data')
        bird_folder_save = os.path.join(data_folder, bird_id)
        bird_folder_data = os.path.join(data_folder_zeke, bird_id)
    else:
        data_folder = os.path.abspath('/Users/Kai/Documents/UCSD/Research/Gentner group/Raw Data')
        results_folder = os.path.abspath('/Users/Kai/Documents/UCSD/Research/Gentner group/Results/'+model_name)
        repos_folder = os.path.abspath('/Users/Kai/Documents/UCSD/Research/Gentner group/Code')
        bird_folder_save = os.path.join(data_folder, bird_id)
        bird_folder_data = bird_folder_save
    
    if not os.path.exists(bird_folder_save):
        os.makedirs(bird_folder_save)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    sys.path.append(os.path.join(repos_folder,'swissknife'))
    from streamtools import spectral as sp
    
    return bird_folder_data, bird_folder_save, repos_folder, results_folder


# # Fuctions Borrowed from Tim

# In[4]:


# Most of the Spectrograms and Inversion are taken from: https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    window_step : int
        Step size between windows
    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))

    ws = window_size
    ss = window_step
    a = X

    valid = len(a) - ws
    nw = int((valid) // ss)
    out = np.ndarray((nw,ws),dtype = a.dtype)

    for i in xrange(nw):
        # "slide" the window along the samples
        start = i * ss
        stop = start + ws
        out[i] = a[start : stop]

    return out


def stft(X, fftsize=128, step=65, mean_normalize=True, real=False,
         compute_onesided=True):
    """
    Compute STFT for 1D real valued input X
    """
    if real:
        local_fft = np.fft.rfft
        cut = -1
    else:
        local_fft = np.fft.fft
        cut = None
    if compute_onesided:
        cut = fftsize // 2
    if mean_normalize:
        X -= X.mean()

    X = overlap(X, fftsize, step)
    
    size = fftsize
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    X = X * win[None]
    X = local_fft(X)[:, :cut]
    return X

def pretty_spectrogram(d,log = True, thresh= 5, fft_size = 512, step_size = 64):
    """
    creates a spectrogram
    log: take the log of the spectrgram
    thresh: threshold minimum power for log spectrogram
    """
    specgram = np.abs(stft(d, fftsize=fft_size, step=step_size, real=False,
        compute_onesided=True))
  
    if log == True:
        specgram /= specgram.max() # volume normalize to max 1
        specgram = np.log10(specgram) # take log
        specgram[specgram < -thresh] = -thresh # set anything less than the threshold as the threshold
    else:
        specgram[specgram < thresh] = thresh # set anything less than the threshold as the threshold
    
    return specgram

# Also mostly modified or taken from https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
def invert_pretty_spectrogram(X_s, log = True, fft_size = 512, step_size = 512/4, n_iter = 10):
    
    if log == True:
        X_s = np.power(10, X_s)

    X_s = np.concatenate([X_s, X_s[:, ::-1]], axis=1)
    X_t = iterate_invert_spectrogram(X_s, fft_size, step_size, n_iter=n_iter)
    return X_t

def iterate_invert_spectrogram(X_s, fftsize, step, n_iter=10, verbose=False):
    """
    Under MSR-LA License
    Based on MATLAB implementation from Spectrogram Inversion Toolbox
    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.
    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.
    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    reg = np.max(X_s) / 1E8
    X_best = copy.deepcopy(X_s)
    for i in range(n_iter):
        if verbose:
            print("Runnning iter %i" % i)
        if i == 0:
            X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                                     set_zero_phase=True)
        else:
            # Calculate offset was False in the MATLAB version
            # but in mine it massively improves the result
            # Possible bug in my impl?
            X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                                     set_zero_phase=False)
        est = stft(X_t, fftsize=fftsize, step=step, compute_onesided=False)
        phase = est / np.maximum(reg, np.abs(est))
        X_best = X_s * phase[:len(X_s)]
    X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                             set_zero_phase=False)
    return np.real(X_t)

def invert_spectrogram(X_s, step, calculate_offset=True, set_zero_phase=True):
    """
    Under MSR-LA License
    Based on MATLAB implementation from Spectrogram Inversion Toolbox
    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.
    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.
    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    size = int(X_s.shape[1] // 2)
    wave = np.zeros((X_s.shape[0] * step + size))
    # Getting overflow warnings with 32 bit...
    wave = wave.astype('float64')
    total_windowing_sum = np.zeros((X_s.shape[0] * step + size))
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))

    est_start = int(size // 2) - 1
    est_end = est_start + size
    for i in range(X_s.shape[0]):
        wave_start = int(step * i)
        wave_end = wave_start + size
        if set_zero_phase:
            spectral_slice = X_s[i].real + 0j
        else:
            # already complex
            spectral_slice = X_s[i]

        # Don't need fftshift due to different impl.
        wave_est = np.real(np.fft.ifft(spectral_slice))[::-1]
        if calculate_offset and i > 0:
            offset_size = size - step
            if offset_size <= 0:
                print("WARNING: Large step size >50% detected! "
                      "This code works best with high overlap - try "
                      "with 75% or greater")
                offset_size = step
            offset = xcorr_offset(wave[wave_start:wave_start + offset_size],
                                  wave_est[est_start:est_start + offset_size])
        else:
            offset = 0
        wave[wave_start:wave_end] += win * wave_est[
            est_start - offset:est_end - offset]
        total_windowing_sum[wave_start:wave_end] += win
    wave = np.real(wave) / (total_windowing_sum + 1E-6)
    return wave

def xcorr_offset(x1, x2):
    """
    Under MSR-LA License
    Based on MATLAB implementation from Spectrogram Inversion Toolbox
    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.
    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.
    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()
    frame_size = len(x2)
    half = frame_size // 2
    corrs = np.convolve(x1.astype('float32'), x2[::-1].astype('float32'))
    corrs[:half] = -1E30
    corrs[-half:] = -1E30
    offset = corrs.argmax() - len(x1)
    return offset

def make_mel(spectrogram, mel_filter, shorten_factor = 1):
    mel_spec =np.transpose(mel_filter).dot(np.transpose(spectrogram))
    mel_spec = scipy.ndimage.zoom(mel_spec.astype('float32'), [1, 1./shorten_factor]).astype('float16')
    mel_spec = mel_spec[:,1:-1] # a little hacky but seemingly needed for clipping 
    return mel_spec

def mel_to_spectrogram(mel_spec, mel_inversion_filter, spec_thresh, shorten_factor):
    """
    takes in an mel spectrogram and returns a normal spectrogram for inversion 
    """
    mel_spec = (mel_spec+spec_thresh)
    uncompressed_spec = np.transpose(np.transpose(mel_spec).dot(mel_inversion_filter))
    uncompressed_spec = scipy.ndimage.zoom(uncompressed_spec.astype('float32'), [1,shorten_factor]).astype('float16')
    uncompressed_spec = uncompressed_spec -4
    return uncompressed_spec

# From https://github.com/jameslyons/python_speech_features

def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.)
    
def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(nfilt=20,nfft=512,samplerate=30000,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.

    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"
    
    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = np.zeros([nfilt,nfft//2])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        if int(bin[j+1]) == int(bin[j+2]):
            fbank[j,int(bin[j+1])] = (int(bin[j+1]) - bin[j]) / (bin[j+1]-bin[j])
        else:
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank

def create_mel_filter(fft_size, n_freq_components = 64, start_freq = 300, end_freq = 8000, samplerate=30000):
    """
    Creates a filter to convolve with the spectrogram to get out mels

    """
    mel_inversion_filter = get_filterbanks(nfilt=n_freq_components, 
                                           nfft=fft_size, samplerate=samplerate, 
                                           lowfreq=start_freq, highfreq=end_freq)
    # Normalize filter
    mel_filter = mel_inversion_filter.T / mel_inversion_filter.sum(axis=1)

    return mel_filter, mel_inversion_filter


# In[5]:


def pretty_spectrogram_zeke(x, s_f, log=True, fft_size=512, step_size=64, window=None,
                       thresh = -5,
                       f_min=0.,
                       f_max=10000.):

    nperseg = fft_size
    noverlap = fft_size - step_size
    f, t, specgram = sg.spectrogram(x, fs=s_f, window=window,
                                       nperseg=nperseg,
                                       noverlap=noverlap,
                                       nfft=None,
                                       detrend='constant',
                                       return_onesided=True,
                                       scaling='density',
                                       axis=-1,
                                       mode='psd')
    if log == True:
        specgram /= specgram.max() # volume normalize to max 1
        specgram = np.log10(specgram) # take log
        specgram[specgram < -thresh] = -thresh # set anything less than the threshold as the threshold
    else:
        specgram[specgram < thresh] = thresh # set anything less than the threshold as the threshold
    
    f_filter = np.where((f > f_min) & (f <= f_max))
    return f[f_filter], t, specgram[f_filter]


# In[6]:


def divide_parts(num_bins, divisor):
    num_bins_each_part_min = num_bins//divisor
    num_bins_each_part_max = num_bins_each_part_min + (num_bins%divisor!=0)
    
    num_bins_each_part = list([num_bins_each_part_max])*int(num_bins%divisor)+list([num_bins_each_part_min])*int(divisor-num_bins%divisor)
    
    return num_bins_each_part

def find_start_end_index(num_list):
    start_index = list()
    end_index = list()
    for i in range(len(num_list)):
        start_index.append(sum(num_list[:i]))
        end_index.append(sum(num_list[:i+1]))
    return start_index, end_index


# In[7]:


def plotspec(spec, vmin=-4, vmax=0):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
    cax = ax.matshow(np.array(spec, dtype='float64'), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower',vmax=vmax, vmin=vmin)
    fig.colorbar(cax)
    #plt.title('Mel Spectrogram')
    return fig


# In[ ]:


def plotspec_auto(spec):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
    cax = ax.matshow(np.array(spec, dtype='float64'), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    fig.colorbar(cax)
    return fig


# In[8]:


def spec2wav(example_spec, mel_inversion_filter, spec_thresh, shorten_factor, fft_size, step_size):
    example_inverted_spectrogram = mel_to_spectrogram(example_spec, mel_inversion_filter,
                                                    spec_thresh=spec_thresh,
                                                    shorten_factor=shorten_factor)
    inverted_mel_audio = invert_pretty_spectrogram(np.transpose(example_inverted_spectrogram), fft_size = 2048,
                                                step_size = 128, log = True, n_iter = 10)
    #plt.plot(inverted_mel_audio)
    return inverted_mel_audio


# In[9]:


def sort_songs(test_output_compiled, test_spec_compiled, num_songs, num_bins, divisor, break_song=False, test_indeces=[0]):
    predicted_song_compiled = list()
    original_song_compiled = list()
    rmse = list()

    if break_song:
        test_indeces = assign_test_indeces(divisor, break_song=break_song)
        num_bins_per_part = divide_parts(num_bins, divisor)
        for song_index in range(num_songs):
            predicted_song = list()
            original_song = list()
            for test_index in test_indeces:
                num_bins_this_index = num_bins_per_part[test_index]
                current_predicted_seg = test_output_compiled[test_index][song_index*num_bins_this_index:(song_index+1)*num_bins_this_index]
                current_original_seg = test_spec_compiled[test_index][song_index*num_bins_this_index:(song_index+1)*num_bins_this_index]
                predicted_song += list(current_predicted_seg)
                original_song += list(current_original_seg)
            rmse.append(np.sqrt(np.mean(np.square(np.array(predicted_song)-np.array(original_song)))))
            predicted_song_compiled.append(predicted_song)
            original_song_compiled.append(original_song)
    else:
        num_songs_each_part = divide_parts(num_songs, divisor)
        if len(test_indeces)>1:
            start_index, end_index = find_start_end_index(num_songs_each_part)
            for test_index in test_indeces:
                for song_index in range(num_songs_each_part[test_index]):
                    #offset = start_index[test_index]*num_bins
                    
                    predicted_song = test_output_compiled[test_index][song_index*num_bins:(song_index+1)*num_bins]
                    predicted_song_compiled.append(predicted_song)
                    original_song = test_spec_compiled[test_index][song_index*num_bins:(song_index+1)*num_bins]
                    original_song_compiled.append(original_song)
                    rmse.append(np.sqrt(np.mean(np.square(np.array(predicted_song)-np.array(original_song)))))
        else:
            test_index=test_indeces
            for song_index in range(num_songs_each_part[test_index[0]]):
                predicted_song = test_output_compiled[0][song_index*num_bins:(song_index+1)*num_bins]
                predicted_song_compiled.append(predicted_song)
                original_song = test_spec_compiled[0][song_index*num_bins:(song_index+1)*num_bins]
                original_song_compiled.append(original_song)
                rmse.append(np.sqrt(np.mean(np.square(np.array(predicted_song)-np.array(original_song)))))
    print([max(rmse),min(rmse)])
    
    return predicted_song_compiled, original_song_compiled, rmse

def assign_test_indeces(divisor, break_song=False):
    if break_song:
        test_indeces = range(divisor)
    else:
        test_indeces = [random.randint(0, divisor-1)]
    return test_indeces


# In[ ]:


def match_length(x, y):
    minlength = min(x.shape[1], y.shape[1])
    x = x[:,:minlength]
    y = y[:,:minlength]
    return(x, y)


# In[10]:


def eval_performance(x, y=None, mode = 'control_w', fft_size = 2048, step_size = 128, log = True, thresh = 5, shorten_factor = 1, 
                     n_mel_freq_components = 64, start_freq = 200, end_freq = 15000, matric = 'corr', output_all = False):

#mode should be:
#'self': evaluating variations among x. (positive control)
#'control_w': evaluating white noise performance against x. (negative control)
#'control_f': evaluating noise (randomly flip signs of audio wave) performance against x. (negative control)
#'paired': x value already includes both x and y arrays, and paired.
#when there's a y value, mode is ignored.

#matric should be:
#'rmse': root mean square error
#'corr': correlation coefficients between spectrograms
#'both': both matrices above

#output_all:
#if False: outputs mean and standard deviation of the matric specified above
#if True: outputs all numbers of the matric above

    if not y and not mode:
        raise ValueError('Either give a y or specify a mode.')
    #if mode and mode is not 'control' and mode is not 'self':
    #    raise ValueError('Mode should be control or self.')
    if y and mode:
        mode = None
        print('Mode has been ignored due to the presence of y values.')
    
    
    if matric is not 'corr' and matric is not 'rmse' and matric is not 'both':
        raise ValueError('mode should be set to corr or rmse or both.')
        
    coeff = list()
    rmse = list()
    mel_filter, mel_inversion_filter = create_mel_filter(fft_size = fft_size,
                                                        n_freq_components = n_mel_freq_components,
                                                        start_freq = start_freq,
                                                        end_freq = end_freq)
    if mode == 'self':
        xnew = list()
        ynew = list()
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                xnew.append(x[i])
                ynew.append(x[j])
        x = xnew
        y = ynew
        
    for i in range(len(x)):
        #plt.plot(original)
        if mode=='paired':
            xi = np.array(x[i][0])
            yi = np.array(x[i][1])
            xi, yi = match_length(xi, yi)
            xi = xi.flatten()
            yi = yi.flatten()
        else:
            xi = np.array(x[i]).flatten()
            if not y:
                if mode == 'control_w':
                    #white noise generation: 1) invert spectrogram back to audio wave. 2) generate a random audio wave
                    #with on normal distribution with the same standard deviation and mean at 0. 3) make a spectrogram of
                    #the noise wave.
                    if x[i].shape[1]!=64:
                        original_audio = invert_pretty_spectrogram(x[i].transpose(), fft_size = fft_size,
                                                    step_size = step_size, log = True, n_iter = 10)
                    else:
                        original_audio = spec2wav(np.array(x[i]).transpose(), mel_inversion_filter, thresh, shorten_factor, fft_size, step_size)
                    #print(len(original_audio))
                    control_audio = np.random.normal(0, np.std(original_audio), len(original_audio))
                    #plt.plot(control_audio)
                    yi = pretty_spectrogram(control_audio.astype('float64'), fft_size = fft_size, step_size = step_size, log = log, 
                                                       thresh = thresh)
                    if x[i].shape[1]!=64:
                        yi = yi[1:-1,:].flatten()
                    else:
                        yi = make_mel(yi, mel_filter, shorten_factor = shorten_factor).flatten()

                if mode == 'control_f':
                    original_audio = spec2wav(np.array(x[i]).transpose(), mel_inversion_filter, thresh, shorten_factor, fft_size, step_size)
                    control_audio = np.array([random.choice([1, -1])*wavbit for wavbit in original_audio])
                    yi = pretty_spectrogram(control_audio.astype('float64'), fft_size = fft_size, step_size = step_size, log = log, 
                                                       thresh = thresh)
                    yi = make_mel(yi, mel_filter, shorten_factor = shorten_factor).flatten()

            else:
                yi = np.array(y[i]).flatten()
     
        if matric == 'corr':
            coeff.append(np.corrcoef([xi, yi])[0][1])
        if matric == 'rmse':
            rmse.append(np.sqrt(np.mean(np.square(np.array(xi)-np.array(yi)))))
        if matric == 'both':
            coeff.append(np.corrcoef([xi, yi])[0][1])
            rmse.append(np.sqrt(np.mean(np.square(np.array(xi)-np.array(yi)))))
        
    if output_all:
        if matric == 'corr':
            return(coeff)
        if matric == 'rmse':
            return(rmse)
        if matric =='both':
            return([coeff, rmse]) 
    else:
        if matric == 'corr':
            return([round(np.mean(coeff, axis=0),4), round(np.std(coeff, axis=0),4)])
        if matric == 'rmse':
            return([round(np.mean(rmse, axis=0),4), round(np.std(rmse, axis=0),4)])
        if matric =='both':
            return([round(np.mean(rmse, axis=0),4), round(np.std(rmse, axis=0),4),
                   round(np.mean(coeff, axis=0),4), round(np.std(coeff, axis=0),4)])


# In[ ]:


def normalize_matrix(matrix, max_val, min_val):
    matrix-=min_val
    matrix/=(max_val-min_val)
    return(matrix)


# In[ ]:


def normalize_list(A, B=None):
    #normalizes either one or two matrix lists (for normalizing original&predicted specs at the same time)
    one_list = False
    if not B:
        B = A
        one_list = True
    #normalize both lists of matrices based on the absolute max and min of both lists, so the internal variation of
    #this entire dataset is maintained.
    max_val = np.max([A])
    min_val = np.min([A])
    newA = list()
    newB = list()
    for matrixA in A:
        newA.append(normalize_matrix(matrixA, max_val, min_val))
    for matrixB in B:
        newB.append(normalize_matrix(matrixB, max_val, min_val))
    if one_list:
        return(newA)
    else:
        return(newA, newB)


# In[11]:


def save_wav(waveform, rate, name, amp_rate=30000):
    waveform_adjusted = np.asarray(waveform/max(waveform)*amp_rate, dtype=np.int16)
    scipy.io.wavfile.write(name, rate, waveform_adjusted)


# In[ ]:


def generate_blanks(songtuples, song_length, ratio = 1):
    #ratio = # of blanks / # of motifs
    
    num_blanks = int(len(songtuples)*ratio)
    motif_starts = np.array(songtuples)[:, 1]
    motif_recs = np.array(songtuples)[:, 0]
    
    max_rec = np.max(motif_recs)+1
    motif_starts_per_rec = list()
    blank_recs = list()
    blank_starts = list()
    
    max_t_per_rec = list()
    for rec in range(max_rec):
        this_rec = [motif_starts[i] for i in range(len(motif_starts)) if motif_recs[i] == rec]
        if not this_rec:
            motif_starts_per_rec.append([])
            max_t_per_rec.append(0)
            continue
        #print(rec, this_rec)
        motif_starts_per_rec.append(this_rec)
        max_t_per_rec.append(np.max(this_rec))
    
    #print(max_t_per_rec)
    for blank_i in range(num_blanks):
        rand_rec = random.randint(0, max_rec-1)
        while not max_t_per_rec[rand_rec]:
            rand_rec = random.randint(0, max_rec-1)
            
        #print(rand_rec)
        this_rec = motif_starts_per_rec[rand_rec]
        this_rec_period = sum([range(motif_start,motif_start+song_length) for motif_start in motif_starts], [])
        this_max = max_t_per_rec[rand_rec]
        rand_start = random.randint(0, this_max-song_length)
        
        while rand_start in this_rec_period:
            rand_start+=song_length
            if rand_start>this_max-song_length:
                rand_start = random.randint(0, this_max-song_length)
        blank_recs.append(rand_rec)
        blank_starts.append(rand_start)
    
    blank_tuples = zip(blank_recs, blank_starts)
    
    return blank_tuples


# In[ ]:


def wav2max(wav_data_rec, song_start, song_end, freq, spec_thresh, fft_size, step_size):
    
    wav_data = wav_data_rec[song_start:song_end+fft_size-step_size]
    _, _, specs = pretty_spectrogram_zeke(wav_data.astype('float64'), freq, thresh = spec_thresh, 
                                                            fft_size=fft_size, log=False, step_size=step_size, 
                                                            window=('gaussian', 80), f_min=0, f_max=15000)
    maxbins = np.argmax(specs, axis = 0)

    max_vals = [specs.transpose()[ind] for ind in list(enumerate(maxbins))]
    #print(max_vals)
    log_vals = [20*math.log10(abs(max_val)) for max_val in max_vals]
    search_array = np.array(log_vals)
    
    #plt.plot(log_vals)
    max_t = np.argmax(search_array)
    log_max = search_array[max_t]
    
    return search_array, max_t, log_max


# In[ ]:


def find_syllable_boundaries(wav_data_rec, song_start, song_end, freq, fft_size, step_size, count, fig_folder_save, spec_thresh=5, 
                             peak_var = 0.6, noise_thresh=47, mode='fft'):
    
    this_wav = wav_data_rec[song_start:song_end]
    env = [np.max(abs(np.array(this_wav[index*step_size:(index+1)*step_size]))) for index in range(int(len(this_wav)/step_size))]
    
    if mode == 'fft':
        offset_bins = 0
        endset_bins = 0
        search_array, max_t, log_max = wav2max(wav_data_rec, song_start, song_end, freq, spec_thresh, fft_size, step_size)

        #if a syllable is clipped at the beginning or the end, extend the song to include the full syllable
        while search_array[0]>noise_thresh:
            offset_bins += int(512//step_size)

            if offset_bins*step_size>12000:
                raise ValueError('Start shifted too much')
            offset_pts = offset_bins*step_size
            song_start -= offset_pts
            search_array, max_t, log_max = wav2max(wav_data_rec, song_start, song_end, freq, spec_thresh, fft_size, step_size)
        if offset_bins:
            print('Start shifted by %d bins' %(offset_bins))

        while search_array[-1]>noise_thresh:
            endset_bins += int(512//step_size)
            if endset_bins*step_size>12000:
                raise ValueError('End shifted too much')
            endset_pts = endset_bins*step_size
            song_end += endset_pts
            search_array, max_t, log_max = wav2max(wav_data_rec, song_start, song_end, freq, spec_thresh, fft_size, step_size)

        heads = list()
        tails = list()
        lengths = list()
        plt.plot(search_array)
        #print(max_t)
        while search_array[max_t]>log_max*peak_var:

            tail_seg = np.array(search_array[max_t:max_t+400])
            tail_t = max_t+np.argmax(tail_seg<noise_thresh)
            #print(tail_t)
            '''
            if search_array[tail_t]>=noise_thresh or max_t==tail_t:
                head_seg = np.array(search_array[max_t:max_t-400:-1])
                try:
                    head_t = max_t-np.argmax(head_seg<noise_thresh)
                except ValueError:
                    break
                #print(head_t)
                search_array[head_t:tail_t+1]=0
                #plt.plot(search_array)

                #if max_t==tail_t:
                #    break
                max_t = np.argmax(search_array)
                #print(max_t)
                if search_array[max_t]<=log_max-peak_var:
                    print('While loop finishes as expected')
                continue
            '''
            head_seg = np.array(search_array[max_t:max(max_t-400, 0):-1])
            #print(head_seg)
            '''
            if not len(head_seg):
                head_seg = np.array(search_array[max_t:0:-1])
            '''
            head_t = max_t-np.argmax(head_seg<noise_thresh)
            if (tail_t-head_t)*step_size<600:
                search_array[head_t:tail_t+1] = 0
                max_t = np.argmax(search_array)
                if search_array[max_t]<=log_max*peak_var:
                    print('While loop finishes as expected')
                continue

            heads.append(head_t-offset_bins)
            tails.append(tail_t-offset_bins)
            #print(head_t)

            lengths.append(tail_t-head_t)

            search_array[head_t:tail_t+1] = 0
            max_t = np.argmax(search_array)

            if search_array[max_t]<=log_max*peak_var:
                print('While loop finishes as expected')

            #print(max_t)

        print(str(len(heads))+' syllables found.')
        print('-'*50)
        heads = sorted(heads)
        tails = sorted(tails)
    
    elif mode == 'wav':
        counts, bins = np.histogram(abs(np.array(wav_data_rec)), bins = 20)
        thresh = bins[np.argmax(counts)+1]
        on_off = [abs(point)>thresh for point in env]
        tails = [index for index in range(len(on_off)) if (on_off[index]==1 and index== len(on_off)-1) or 
                 (on_off[index]==1 and on_off[index+1]==0)]
        heads = [index for index in range(len(on_off)) if (on_off[index]==1 and index== 0) or 
                 (on_off[index]==1 and on_off[index-1]==0)]
        lengths = [tail-head for head, tail in zip(heads, tails)]
        
        testheads = [heads[i] for i in range(len(heads)) if i ==0             or (heads[i]-tails[i-1])*step_size>600]
        testtails = [tails[i] for i in range(len(tails)) if i == len(tails)-1 or (heads[i+1]-tails[i])*step_size>600]
        testlengths = [tail-head for head, tail in zip(testheads, testtails)]
        newheads = [testheads[i] for i in range(len(testheads)) if testlengths[i]*step_size>300]
        newtails = [testtails[i] for i in range(len(testtails)) if testlengths[i]*step_size>300]
        
        heads = newheads
        tails = newtails
        lengths = [tail-head for head, tail in zip(heads, tails)]
        
        durations = []
        for head, tail in zip(heads, tails):
            durations+=range(head, tail)
        new_on_off = [(i in durations) for i in range(len(on_off))]
        on_off = new_on_off
        
    elif mode == 'sigma':
        window_size = freq/1000*40
        counts, bins = np.histogram(abs(np.array(wav_data_rec)), bins = 20)
        thresh = np.mean(abs(motif_data))+np.std(abs(np.array(motif_data)))*2
        
    fig = plt.figure()
    plt.plot(on_off)
    plt.plot(np.array(env)/np.max(env))
    plt.title(str(count))
    plt.savefig(os.path.join(fig_folder_save, '%03d_onoff.png' %(count)))
    plt.close(fig)
    
    return [heads, tails, lengths]


# In[ ]:


def bins2wav(i_bin, bin_size):
    if len(i_bin)>1:
        i_wav = [index*int(bin_size) for index in i_bin]
    else:
        i_wav = i_bin*int(bin_size)
    return i_wav


# In[ ]:


def make_specs(wav_file, rec, song_ideal_start, song_ideal_end, recording_folder_save, bin_size, n_mel_freq_components, 
              start_freq, end_freq, wav_rate = 30000):
    
    fft_size = cal_fft_size(bin_size) # window size for the FFT
    step_size = bin_size # distance to slide along the window (in time)
    spec_thresh = 5 # threshold for spectrograms (lower filters out more noise)
    lowcut = 500 # Hz # Low cut for our butter bandpass filter
    highcut = 14999 # Hz # High cut for our butter bandpass filter
    # For mels
    shorten_factor = 1 # how much should we compress the x-axis (time)
    
    if type(wav_file)==str:
        wav_rate, wav_data = wavfile.read(wav_file)
        wav_data = wav_data[song_ideal_start:song_ideal_end]
    else:
        wav_data = wav_file[:]
    wav_data = butter_bandpass_filter(wav_data, lowcut, highcut, wav_rate, order=1)
                
    #plot each filtered unmel-edwav
    fig = plt.figure()
    plt.plot(wav_data)
    plt.title(str(rec)+': '+str(song_ideal_start))
    plt.savefig(os.path.join(recording_folder_save, 'motif '+str(rec)+' '+str(song_ideal_start)+'.png'))
    plt.close(fig)
                
    #mel and compile into a list based on time bins
    _, _, wav_spectrogram = pretty_spectrogram_zeke(wav_data.astype('float64'), wav_rate, thresh = spec_thresh, 
                                                            fft_size=fft_size, log=True, step_size=step_size, 
                                                            window=('gaussian', 80), f_min=0, f_max=15000)
    #print(wav_spectrogram)
    
    mel_filter, mel_inversion_filter = create_mel_filter(fft_size = fft_size,
                                                        n_freq_components = n_mel_freq_components,
                                                        start_freq = start_freq,
                                                        end_freq = end_freq)
    mel_array = make_mel(wav_spectrogram.transpose(), mel_filter, shorten_factor = shorten_factor).transpose()
    #print(mel_array.shape)
    mel_list = list(mel_array)
    
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(8,4))
    cax = ax.matshow(np.array(mel_list, dtype = 'float64').transpose(), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    fig.colorbar(cax)
    plt.savefig(os.path.join(recording_folder_save, 'motif '+str(rec)+' '+str(song_ideal_start)+'_mel.png'))
    plt.close(fig)
    
    return mel_list


# In[ ]:


def spike_counter(current_t_list, current_cluster_list, neural_ideal_start, neural_ideal_end, song_ideal_start, rec,
                  recording_folder_save,
                  num_clusters, bin_size, num_bins, num_lookbacks, plot_raster=True):
    
    neural_start_index = bisect.bisect_left(current_t_list, neural_ideal_start)-1 #locate the index of starting point of the current motif
    neural_start_val = current_t_list[neural_start_index] #the actual starting point of the current motif
    #start_empty_bins = (neural_start_val-neural_ideal_start)//bin_size
                
    neural_end_index = bisect.bisect_left(current_t_list, neural_ideal_end)-1 #the end index of the current motif wrt indexing within the entire recording
    #end_empty_bins = (neural_ideal_end-current_t_list[neural_end_index])//bin_size

    t_during_song = current_t_list[neural_start_index:neural_end_index+1]-neural_ideal_start#normalize time sequence
    cluster_during_song = current_cluster_list[neural_start_index:neural_end_index+1] #extract cluster sequence

    counts_list = list() #contains all counts data for this motif
    for i in range(num_bins+num_lookbacks):
        bin_start_index = bisect.bisect_left(t_during_song, i*bin_size)
        bin_end_index = bisect.bisect_left(t_during_song,(i+1)*bin_size)
        counts, bins = np.histogram(cluster_during_song[bin_start_index:bin_end_index], bins=np.arange(0,num_clusters+1))
        counts_list.append(counts)
    
    if plot_raster:
        
        fig = plt.figure()
        plt.plot(np.sum(np.array(counts_list), axis=1))
        plt.title(str(rec)+': '+str(song_ideal_start))
        plt.savefig(os.path.join(recording_folder_save, 'motif '+str(rec)+' '+str(song_ideal_start)+'_neural.png'))
        plt.close(fig)
        
        fig = plt.imshow(np.array(counts_list).transpose())
        plt.title(str(rec)+': '+str(song_ideal_start))
        plt.savefig(os.path.join(recording_folder_save, 'motif '+str(rec)+' '+str(song_ideal_start)+'_raster.png'))
        plt.close('all')
        
    return counts_list


# In[ ]:


def prep_raster(current_t_list, current_cluster_list, song_start, adjusted_song_length, positions,
                num_clusters, save_folder):
    #current_t_list is absolute time within the recording
    #current_cluster_list indicate which cluster fired at that time
    #song_start is the absolute time when song starts
    #positions is a preset list of lists with each sublist correspond to a cluster
    
    start_index = bisect.bisect_left(current_t_list, song_start)
    end_index = bisect.bisect_left(current_t_list, song_start+adjusted_song_length)
    t_list = current_t_list[start_index:end_index]
    this_time = [int(t-song_start) for t in t_list]
    this_cluster = current_cluster_list[start_index:end_index]
    for c in range(num_clusters):
        positions[c].append(np.array([this_time[i] for i, x in enumerate(this_cluster) if x == c], dtype = 'double'))
    
    np.save(os.path.join(save_folder, 'positions.npy'), positions)
    return positions
    


# In[ ]:


def raster_plotter(positions, num_clusters, recording_folder_save):
    for c in range(num_clusters):
        i = 1
        while not positions[c][0].size:
            positions[c][0], positions[c][i] = positions[c][i], positions[c][0]
            i += 1
            if i==61:
                positions[c][0] = np.array([0])
        plt.figure(figsize=(20,10))
        plt.eventplot(np.array(positions[c]))
        plt.title('cluster '+str(c))
        plt.savefig(os.path.join(recording_folder_save, '%02d_raster.png' %(c)))
        plt.close('all')


# In[ ]:


def separate_recs(directory, neural_file_name, bird_id, model_name):
    bird_folder_data, bird_folder_save, repos_folder, results_folder = locate_folders(bird_id, model_name)
    kwik_file = os.path.join(directory, neural_file_name)
    
    with h5py.File(kwik_file,'r') as f:
        time_samples = f['/channel_groups/0/spikes/time_samples'].value
        cluster_num = f['/channel_groups/0/spikes/clusters/main'].value
        recording_num = f['/channel_groups/0/spikes/recording'].value
        cont_test = np.array(recording_num[:-1], dtype='int64')-np.array(recording_num[1:], dtype='int64')
        errors = [i for i, j in enumerate(cont_test) if j>0]
        if errors:
            for error_i in errors:
                if time_samples[error_i]<time_samples[error_i+1]:
                    recording_num[error_i+1] = recording_num[error_i]
                else:
                    recording_num[error_i+1] = recording_num[error_i+2]
            print('***Erroneous indices fixed.')
            
        print('Start working on '+directory)        
        rec_counter = collections.Counter(recording_num)
        rec_list = rec_counter.keys()
        spike_counts = rec_counter.values()
        time_samples_each_rec = list()
        cluster_each_rec = list()

        #separate time samples based on recording
        for rec in rec_list:
            prev_num_samples = sum(spike_counts[:rec])
            time_samples_each_rec.append(time_samples[prev_num_samples:prev_num_samples+spike_counts[rec]])
            cluster_each_rec.append(cluster_num[prev_num_samples:prev_num_samples+spike_counts[rec]])
            
    return time_samples_each_rec, cluster_each_rec


# In[ ]:


def pattern_warp(current_t_list, current_cluster_list, neural_ideal_start, neural_ideal_end, song_ideal_start, rec,
                  recording_folder_save,
                  num_clusters, bin_size, num_bins, num_lookbacks, plot_raster=True):
    
    #needed input: 
    #template_heads_dps: list of start times of each syllable within the song template !!NO LOOKBACK INFO IN TEMPLATE
    #template_tails_dps:           end
    #test_heads_dps:             start                                   test song
    #test_tails_dps:               end                      
    #num_lookbacks: lookback bins
    #bin_size: normal bin size used in template
    #neural_ideal_start: already taken into account lookback bins
    #song_ideal_start
    #song_ideal_end
    
    if len(template_heads_dps)!=len(test_heads_dps):
        raise ValueError('Test set has more or fewer syllables than template!')
    
    #shift to include lookback periods
    for i in range(len(template_heads_dps)):
        template_heads_dps[i] += bin_size*num_lookbacks
        template_tails_dps[i] += bin_size*num_lookbacks
        test_heads_dps[i] += bin_size*num_lookbacks
        test_tails_dps[i] += bin_size*num_lookbacks
    
    #calculate start (and end offsets)
    start_offset_dps = test_heads_dps[0]-template_heads_dps[0]
    #end_offset_dps = test_tails_dps[-1]-template_tails_dps[-1]
    
    #length of each syllable and silence
    temp_syllable_lengths = [temp_tail-temp_head for temp_head, temp_tail in zip(template_heads_dps, template_tails_dps)]
    test_syllable_lengths = [test_tail-test_head for test_head, test_tail in zip(test_heads_dps, test_tails_dps)]
    
    temp_silence_lengths = [temp_head-temp_tail for temp_head, temp_tail in zip(template_heads_dps[1:], template_tails_dps[:-1])]
    test_silence_lengths = [test_head-test_tail for test_head, test_tail in zip(test_heads_dps[1:], test_tails_dps[:-1])]
    
    bin_size_ratio = [test_l/temp_l for temp_l, test_l in zip(temp_syllable_lengths, test_syllable_lengths)]
    #test_bin_sizes = [ratio*bin_size for ratio in bin_size_ratio]
    
    song_actual_start_dps = song_ideal_start+start_offset_dps
    neural_actual_start_dps = neural_ideal_start+start_offset_dps
    
    #song_actual_end_dps = song_ideal_end+end_offset_dps
    
    #lookback bins start/end points
    bin_start_dps = [i*bin_size for i in range(num_lookbacks)]
    bin_end_dps = [(i+1)*bin_size for i in range(num_lookbacks)]
    
    last_end_dps = bin_end_dps[-1]
    
    #silence at the beginning
    bin_start_dps.append(range(last_end_dps, ))
    
        
    #head_offsets = [test_head-temp_head for test_head, temp_head in zip(test_heads_dps, template_heads_dps)]
    #tail_offsets = [test_head-temp_head for test_head, temp_head in zip(test_heads_dps, template_heads_dps)]
    
    neural_start_index = bisect.bisect_left(current_t_list, neural_actual_start_dps)-1 #locate the index of starting point of the current motif
    #start_empty_bins = (neural_start_val-neural_ideal_start)//bin_size
                
    neural_end_index = bisect.bisect_left(current_t_list, song_actual_end_dps) #the end index of the current motif wrt indexing within the entire recording
    #end_empty_bins = (neural_ideal_end-current_t_list[neural_end_index])//bin_size

    t_during_song_dps = current_t_list[neural_start_index:neural_end_index]-neural_actual_start_dps#normalize time sequence
    cluster_during_song = current_cluster_list[neural_start_index:neural_end_index] #extract cluster sequence

    counts_list = list() #contains all counts data for this motif
    for i in range(num_bins+num_lookbacks):
        bin_start_index = bisect.bisect_left(t_during_song, i*bin_size)
        bin_end_index = bisect.bisect_left(t_during_song,(i+1)*bin_size)
        counts, bins = np.histogram(cluster_during_song[bin_start_index:bin_end_index], bins=np.arange(0,num_clusters+1))
        counts_list.append(counts)
    
    if plot_raster:
        fig = plt.figure()
        plt.plot(np.sum(np.array(counts_list), axis=1))
        plt.title(str(rec)+': '+str(song_ideal_start))
        plt.savefig(os.path.join(recording_folder_save, 'motif '+str(rec)+' '+str(song_ideal_start)+'_neural.png'))
        plt.close(fig)
        
    return counts_list


# In[ ]:


def which_dir(bird_folder_data, specify_subdir):
    if specify_subdir:
        subdirs = [os.path.join(bird_folder_data, specify_subdir)]
    else:
        subdirs = [combined[0] for combined in os.walk(bird_folder_data)][1:]
    return subdirs


# In[ ]:


def cal_modified_length(song_length, bin_size, start_extend_bins, end_extend_bins, num_lookbacks, mode='neuro2spec'):
    fft_size = cal_fft_size(bin_size)
    if mode=='neuro2spec':
        extended_song_length = song_length+bin_size*(end_extend_bins-start_extend_bins)+fft_size
        neural_song_length = extended_song_length+(num_lookbacks-2)*bin_size
    elif mode=='spec2neuro':
        extended_song_length = song_length+bin_size*(end_extend_bins-start_extend_bins+num_lookbacks)+fft_size
        neural_song_length = extended_song_length+(-num_lookbacks-2)*bin_size
    #take out two bins due to mel processing algorithm
    
    return extended_song_length, neural_song_length


# In[ ]:


def cal_fft_size(bin_size):
    fft_size = bin_size*16
    return fft_size


# In[ ]:


def cal_extend_size(bin_size, start_extend_bins, end_extend_bins, fft_size = False):
    if not fft_size:
        fft_size = cal_fft_size(bin_size)
    start_extend_bins *= int(fft_size/bin_size)
    end_extend_bins *= int(fft_size/bin_size)
    return start_extend_bins, end_extend_bins


# In[ ]:


def cal_num_bins(song_length, bin_size, start_extend_bins, end_extend_bins):
    num_bins = int(song_length/bin_size-2-start_extend_bins+end_extend_bins)#number of bins in each song
    return num_bins


# In[ ]:


def calculate_neural_mel(directory, time_samples_each_rec, cluster_each_rec, extended_song_length, neural_song_length, 
                         recording_folder_save, slice_shuffle_pattern,
                         neural_file_name, song_file_name, bird_id, model_name, num_clusters, bin_size, num_bins, 
                         num_lookbacks, n_mel_freq_components, start_freq, end_freq, start_extend_bins, 
                         end_extend_bins, mode='neuro2spec', labels=None, flatten_inputs = False, plot_raster = False,
                        warp_method = None):
    if song_file_name.endswith('kwik') or song_file_name.endswith('kwe'):
        song_file_type = 'kwik'
    else:
        song_file_type = 'pickle'
    song_file = os.path.join(directory, song_file_name)
    
    fig_folder_save = os.path.join(recording_folder_save, 'mel', '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now()))
    
    if not os.path.exists(fig_folder_save):
        os.makedirs(fig_folder_save)
        
    mel_neural_comb = list()
    
    if song_file_type=='kwik':
        try:
            motif_starts = h5py.File(song_file, 'r')['event_types/singing/motiff_1/time_samples'].value  #start time of each motif
            motif_recs = h5py.File(song_file, 'r')['event_types/singing/motiff_1/recording'].value  #recording number of each motif
        except:
            motif_starts = h5py.File(song_file, 'r')['event_types/singing/motiff_2/time_samples'].value  #start time of each motif
            motif_recs = h5py.File(song_file, 'r')['event_types/singing/motiff_2/recording'].value  #recording number of each motif
        songtuples = zip(motif_recs, motif_starts)
        
        positions = [[] for i in range(num_clusters)]
        spec_list = list()
        wav_list = list()
        count = 0

        for rec, ideal_start in songtuples:
            mel_neural_comb_song = list()
                
            if mode=='neuro2spec':
                neural_ideal_start = ideal_start+bin_size*(start_extend_bins+1-num_lookbacks)
                neural_ideal_end = neural_ideal_start+neural_song_length

                song_ideal_start = ideal_start+bin_size*start_extend_bins
                song_ideal_end = song_ideal_start+extended_song_length
                song_ideal_end = song_ideal_end+int((16-num_lookbacks)%16)*bin_size
                
            elif mode=='spec2neuro':
                
                song_ideal_start = ideal_start+bin_size*(start_extend_bins-num_lookbacks)
                song_ideal_end = song_ideal_start+extended_song_length
                
                song_ideal_end = song_ideal_end+int((16-num_lookbacks)%16)*bin_size
                
                neural_ideal_start = ideal_start+bin_size*(start_extend_bins+1)
                neural_ideal_end = neural_ideal_start+neural_song_length

                
            #process wav file into mel spec
            wav_name = 'experiment-rec_0%02d.mic.wav' % (rec)
            wav_file = os.path.join(directory, wav_name)
            
            wav_rate, wav_data = wavfile.read(wav_file)
            wav_data = wav_data[song_ideal_start:song_ideal_end]
            wav_list.append(wav_data)
            
            mel_list = make_specs(wav_file, rec, song_ideal_start, song_ideal_end, fig_folder_save, bin_size, n_mel_freq_components, 
                                  start_freq, end_freq)
            
            current_t_list = time_samples_each_rec[rec] #extract neural activities for current motif
            current_cluster_list = cluster_each_rec[rec]  #extract cluster info for current motif
            
            #add here warps
            if warp_method == 'anderson':
                if not count:
                    temp_list = None
                    mel_list, w_binstarts, w_binends = anderson_warp(mel_list, temp_list, fig_folder_save, bin_size, num_lookbacks, 
                                                                        song_ideal_start, count)
                    temp_list = mel_list[:]
                else:
                    mel_list, w_binstarts, w_binends = anderson_warp(mel_list, temp_list, fig_folder_save, bin_size, num_lookbacks, 
                                                                        song_ideal_start, count)
                w_binturns = [w_binstarts, w_binends]
                counts_list = spike_counter_warp(current_t_list, current_cluster_list, w_binturns, rec,
                                              fig_folder_save, count, 
                                              num_clusters, num_bins, num_lookbacks, bin_size, plot_raster=plot_raster)
            else:
                counts_list = spike_counter(current_t_list, current_cluster_list, neural_ideal_start, neural_ideal_end, song_ideal_start, rec,
                                              fig_folder_save,
                                              num_clusters, bin_size, num_bins, num_lookbacks)
            
            if plot_raster:
                if warp_method == 'anderson':
                    start_index = bisect.bisect_left(current_t_list, song_ideal_start)
                    end_index = bisect.bisect_left(current_t_list, song_ideal_end)
                    
                    t_list = current_t_list[start_index:end_index]
                    new_t_list = warp_spike_train(w_binturns, bin_size, t_list, warp_method=warp_method)

                    this_time = [int(t-w_binstarts[0]) for t in new_t_list]
                    this_cluster = current_cluster_list[start_index:end_index]
                    for c in range(num_clusters):
                        positions[c].append(np.array([this_time[i] for i, x in enumerate(this_cluster) if x == c]))
                    
                    np.save(os.path.join(fig_folder_save, 'positions.npy'), positions)
                        
                else:
                    positions = prep_raster(current_t_list, current_cluster_list, song_ideal_start, extended_song_length, positions,
                                            num_clusters, fig_folder_save)
                
            for i in range(num_bins):
                if mode=='neuro2spec':
                    inputs = np.array(counts_list[i:i+num_lookbacks])
                    if flatten_inputs:
                        inputs=inputs.flatten()
                    outputs = np.array(mel_list[i])
                    
                    #if shuffle, then update mel_list
                    if slice_shuffle_pattern:
                        outputs = outputs[np.array(slice_shuffle_pattern[i])]
                        mel_list[i] = outputs[:]
                    mel_neural_tuple = (inputs, outputs)
                    
                    mel_neural_comb_song.append(mel_neural_tuple)
                elif mode=='spec2neuro':
                    inputs = np.array(mel_list[i:i+num_lookbacks])
                    if flatten_inputs:
                        inputs=inputs.flatten()
                    mel_neural_tuple = (inputs, counts_list[i])
                    mel_neural_comb_song.append(mel_neural_tuple)
               
            spec_list.append(np.array(mel_list))
            
            if counts_list[0].shape[0] != num_clusters:
                raise ValueError('check cluster histogram')
            mel_neural_comb.append(mel_neural_comb_song)
            count+=1
        if plot_raster:
            raster_plotter(positions, num_clusters, fig_folder_save)
        spec_file = os.path.join(fig_folder_save, 'specs.npy')
        np.save(spec_file, np.array(spec_list))
        wav_file = os.path.join(fig_folder_save, 'wavs.npy')
        np.save(wav_file, np.array(wav_list))
            
    else:
        try:
            pickle_data = pd.read_pickle(song_file)
        except:
            song_file = os.path.join('/mnt/cube/earneodo/bci_zf/proc_data/', bird_id, song_file_name)
        pickle_data = pd.read_pickle(song_file)
        indeces = [i for i, j in enumerate(pickle_data['recon_folder']) if directory in j]
        indeces = [i for i, j in zip(indeces, pickle_data['syllable_labels'][indeces]) if j in labels]
        total_bins = 0
            
        for ind in indeces:
            mel_neural_comb_song = list()
            wav_file = pickle_data['recon_folder'][ind]
            rec = int(wav_file[wav_file.index('rec_')+4:wav_file.index('rec_')+7])
            ori_song_length = int(pickle_data['recon_length'][ind]*30000)
            song_length, neural_length = cal_modified_length(ori_song_length, bin_size, start_extend_bins=0, end_extend_bins=0, 
                                                                 num_lookbacks=num_lookbacks)
            num_bins = int(np.ceil(ori_song_length/bin_size-2))

            total_bins+=num_bins

            song_ideal_start = int(pickle_data['recon_t_rel_wav'][ind]*30000)
            song_ideal_end = song_ideal_start+song_length
            song_ideal_end = song_ideal_end+int((16-num_lookbacks)%16)*bin_size

            neural_ideal_start = song_ideal_start+bin_size*(1-num_lookbacks)
            neural_ideal_end = neural_ideal_start+neural_song_length

            wav_rate, wav_data = wavfile.read(wav_file)
            wav_data = wav_data[song_ideal_start:song_ideal_end]
            
            mel_list = make_specs(wav_data, wav_rate, rec, song_ideal_start, song_ideal_end, fig_folder_save, bin_size, n_mel_freq_components, 
                                  start_freq, end_freq)

            current_t_list = time_samples_each_rec[rec] #extract neural activities for current motif
            current_cluster_list = cluster_each_rec[rec]  #extract cluster info for current motif

            counts_list = spike_counter(current_t_list, current_cluster_list, neural_ideal_start, neural_ideal_end, song_ideal_start, rec,
                                          fig_folder_save,
                                          num_clusters, bin_size, num_bins, num_lookbacks)

            for i in range(num_bins):
                mel_neural_tuple = (np.array(counts_list[i:i+num_lookbacks]), np.array(mel_list[i]))
                mel_neural_comb_song.append(mel_neural_tuple)
            
            mel_neural_comb.append(mel_neural_comb_song)
                
    return mel_neural_comb


# In[ ]:


def all_to_song_based(all_combs, num_bins):
    
    num_combs = len(all_combs)
    num_songs = int(num_combs/num_bins)
    new_combs = list()
    start_indeces = range(0, num_combs, num_bins)
    if len(start_indeces)!= num_songs:
        raise ValueError('regoup error')
    for index in start_indeces:
        new_combs.append(all_combs[index:index+num_bins])
        
    return new_combs


# In[ ]:


def song_based_to_all(song_based):
    all_combs = list()
    for song in song_based:
        all_combs+=song
    return all_combs


# In[ ]:


def song_based_to_batch_based(song_based, num_bins):
    batch_based = list()
    num_songs = len(song_based)
    for i_bin in range(num_bins):
        this_batch = list()
        for i_song in range(num_songs):
            this_batch.append(song_based[i_song][i_bin])
        #this_batch = np.array(this_batch)
        batch_based.append(this_batch)
    return batch_based


# In[ ]:


def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir,name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def screen_dates(bird_folder_data, song_file_name, bird_id, model_name):
    bird_folder_data, bird_folder_save, repos_folder, results_folder = locate_folders(bird_id, model_name)
    subdirs = get_immediate_subdirectories(bird_folder_data)

    for direct in subdirs:
        song_file = os.path.join(direct, song_file_name)
        try:
            trial1 = h5py.File(song_file,'r')['event_types/singing/motiff_1/time_samples'].value
        except IOError:
            print('--->Skipping '+direct+' because it doesnt contain spiking kwik files.')
            print('*'*60)
            continue
        with h5py.File(song_file, 'r') as f:
            motif_starts = f['event_types/singing/motiff_1/time_samples'].value
            print(direct)
            print(len(motif_starts))
            print('='*60)


# In[ ]:


def make_shuffle_pattern(slice_shuffle_pattern, num_bins, num_features):
    automatic_shuffle = False
    if not isinstance(slice_shuffle_pattern, list) and slice_shuffle_pattern:
        automatic_shuffle = True
        print('Generating shuffling pattern.')
        slice_shuffle_pattern = list()
        for i in range(num_bins):
            arr = range(num_features)
            np.random.shuffle(arr)
            slice_shuffle_pattern.append(arr)
    elif slice_shuffle_pattern:
        print('Shuffling with existing pattern.')
        
    else:
        print('Not Shuffling.')
    
    return slice_shuffle_pattern, automatic_shuffle


# In[ ]:


def cal_warp_path(cost):
    #starting here every index will be shifted by 1 due to padding
    #n should be the template size
    #m should be the sample size
    n = cost.shape[0]
    m = cost.shape[1]
    new_cost = np.zeros([n+1, m+1])
    new_cost[1:, 1:] = cost
    cost = new_cost[:]
    D = np.empty([n+1, m+1])
    D[0,0] = 0
    D[1,1] = cost[0, 0]
    D[0, 1:] = float('inf')
    D[1:, 0] = float('inf')
    D[1, 2:] = float('inf')
    D[2:, 1] = float('inf')
    
    for i in range(2, n+1):
        for j in range(2, m+1):
            if D[i-1, j] == D[i-2, j]+cost[i-1, j]:
                D[i, j] = min(D[i-1, j-1], D[i-1, j-2])+cost[i, j]
            else:
                D[i, j] = min(D[i-1, j-1], D[i-1, j-2], D[i-1, j])+cost[i, j]
                
    steps = list()
    current_j = m
    steps.append(current_j)
    
    for current_i in range(n, 2, -1):
        if D[current_i-1, current_j] == D[current_i-2, current_j]+cost[current_i-1, current_j]:
            current_j = current_j-np.argmin([D[current_i-1, current_j-1], D[current_i-1, current_j-2]])-1
        else:
            current_j = current_j-np.argmin([D[current_i-1, current_j], D[current_i-1, current_j-1], D[current_i-1, current_j-2]])
        steps.append(current_j)
    
#     alt_steps = list()
#     for i in range(n):
#         alt_steps.append(np.argmin(D[i+1,:])-1)
        
    if steps[-1]>3:
        raise ValueError('Warping does not start from beginning')
    steps.append(1)
    steps = [step-1 for step in steps[::-1]]
    #steps = zip(range(len(steps)), [step-1 for step in steps][::-1])
    return(D, steps) 


# In[ ]:


def cal_cost_matrix(samp, temp):
    cost_matrix = np.empty([len(samp), len(temp)])
    for i in range(len(samp)):
        for j in range(len(temp)):
            cost_matrix[i, j] = np.mean(np.square(np.array(samp[i])-np.array(temp[j])))
    return cost_matrix


# In[ ]:


def anderson_warp(samp, temp, fig_folder_save, bin_size, num_lookbacks, song_ideal_start, count):
    #first is the template
    if not temp:
        warped_specs = samp[:]
        w_binstarts = range(-(num_lookbacks-1),len(samp))
        w_binstarts = [start*bin_size+song_ideal_start for start in w_binstarts]
        w_binends = range(-(num_lookbacks-2),len(samp)+1)
        w_binends = [end*bin_size+song_ideal_start for end in w_binends]
    
    else:
        cost = cal_cost_matrix(temp, samp)
        D, steps= cal_warp_path(cost)
        plt.figure()
        plt.matshow(D, origin = 'lower')
        plt.plot(steps)
        plt.savefig(os.path.join(fig_folder_save, '%02d warp path.png' %(count)))
        plt.close('all')
        
        if type(steps[0])==tuple:
            steps = [j for i, j in steps]
        num_bins = len(steps)
        warped_specs = list()
        w_binstarts = list()
        w_binends = list()
        w_binstarts+=range(-(num_lookbacks-1)*bin_size,0, bin_size)
        w_binends += range(-(num_lookbacks-2)*bin_size, bin_size, bin_size) 
        last_j = -1
        for i in range(num_bins):
            current_j = steps[i]
            warped_specs.append(samp[current_j])
            if current_j==last_j:
                w_binends[-1] = int((current_j+0.5)*bin_size)
                this_bin_start = int((current_j+0.5)*bin_size)
                this_bin_end = (current_j+1)*bin_size
            else:
                this_bin_start = current_j*bin_size
                this_bin_end = (current_j+1)*bin_size
                last_j = current_j
            w_binstarts.append(this_bin_start)
            w_binends.append(this_bin_end)
        #print(pd.DataFrame(zip(w_binstarts, w_binends)))
        w_binstarts = [start+song_ideal_start for start in w_binstarts]
        w_binends = [end+song_ideal_start for end in w_binends]
    
    fig = plotspec_auto(np.array(warped_specs).transpose())
    fig.savefig(os.path.join(fig_folder_save, '%02d warped spec.png' %(count)))
    plt.close(fig)
    
    return warped_specs, w_binstarts, w_binends


# # Make Datasets

# ### Vanilla ZF Spec Prediction

# In[5]:


def make_datasets_finch(neural_file_name, song_file_name, song_length, bin_size, num_clusters, num_lookbacks, bird_id, model_name,
                        start_extend_bins=0, end_extend_bins=0, specify_subdir=None,
                        n_mel_freq_components = 64, start_freq = 200, end_freq = 15000, slice_shuffle_pattern=None, 
                        warp_method=None):
    '''
    neural_file_name: kwik file that contains neural data
    song_file_name: kwik file that contains motif starting time and recordings
    song_length: length of motif
    bin_size: how many datapoints to combine into a bin
    num_clusters: how many clusters of neurons are recorded
    start_extend_bins: at the beginning of the motif, how many bins do you want to shift. Shifting the start to the left
                        is negative, and shifting to the right is positive. Default to be 0
    end_extend_bins: same as start_extend_bins, how many bins to shift at the end of motif. Shifting left is negative and
                        right is positive. Default to be 0.
    warp_method: None of not warping, 'anderson' if using warping algorithm published by anderson et al
    '''
    if slice_shuffle_pattern and (start_extend_bins or end_extend_bins):
        raise ValueError('Slice shuffling algorithm does not support extend bins now.')
    print('Current warping method is: '+str(warp_method))
    
    bird_folder_data, bird_folder_save, repos_folder, results_folder = locate_folders(bird_id, model_name)
    save_name = 'mel_%02d_%d_%d_lstm.p' %(num_lookbacks, start_extend_bins, end_extend_bins)
    
    start_extend_bins, end_extend_bins = cal_extend_size(bin_size, start_extend_bins, end_extend_bins)
    num_bins = cal_num_bins(song_length, bin_size, start_extend_bins, end_extend_bins)
    subdirs = which_dir(bird_folder_data, specify_subdir)
    extended_song_length, neural_song_length = cal_modified_length(song_length, bin_size, start_extend_bins, end_extend_bins, num_lookbacks) 
    
    '''
    all_batches = list()
    for _ in range(num_bins):
        all_batches.append(list())
    '''
    mel_neural_comb=list()
    
    slice_shuffle_pattern, automatic_shuffle = make_shuffle_pattern(slice_shuffle_pattern, num_bins, n_mel_freq_components)
    
    for directory in subdirs:
        kwik_file = os.path.join(directory, neural_file_name)
        song_file = os.path.join(directory, song_file_name)
        
        try:
            trial1 = h5py.File(kwik_file,'r')['/channel_groups/0/spikes/time_samples'].value
        except IOError:
            print('--->Skipping '+directory+' because it doesnt contain spiking kwik files.')
            continue
            
        if "sleep" in directory:
            print('--->Skipping '+directory+' because it is a sleep recording.')
            continue
        
        recording = directory[directory.index('day-'):]
        recording_folder_save = os.path.join(bird_folder_save, recording)
        if not os.path.exists(recording_folder_save):
            os.makedirs(recording_folder_save)
        
        time_samples_each_rec, cluster_each_rec = separate_recs(directory, neural_file_name, bird_id, model_name)
            
        mel_neural_comb_dir = calculate_neural_mel(directory, time_samples_each_rec, cluster_each_rec, extended_song_length, neural_song_length, 
                                                     recording_folder_save, slice_shuffle_pattern,
                                                     neural_file_name, song_file_name, bird_id, model_name, num_clusters, bin_size, num_bins, 
                                                     num_lookbacks, n_mel_freq_components, start_freq, end_freq, start_extend_bins, 
                                                     end_extend_bins, warp_method = warp_method)
        
        mel_neural_comb += mel_neural_comb_dir
        #mel_neural_comb is a collection of lists, each inidividual list is a song, contains a colletion of neural-mel 
        #tuples within a song
        
        print('Finished.')
        
    mel_neural_comb = song_based_to_all(mel_neural_comb)
        
    #regoup into batch based   
    pickle.dump(mel_neural_comb, open(os.path.join(bird_folder_save, save_name),'wb'))
    
    if automatic_shuffle:
        return mel_neural_comb, slice_shuffle_pattern
    else:
        return mel_neural_comb


# ### Spec Prediction with Spec

# In[ ]:


def make_datasets_spec2spec(song_file_name, song_length, bin_size, num_clusters, num_lookbacks, bird_id, model_name, 
                            start_extend_bins=0, end_extend_bins=0, specify_subdir=None,
                            n_mel_freq_components = 64, start_freq = 200, end_freq = 15000):
    '''
    This function uses spectrogram to predict spectrogram.
    song_file_name: kwik file that contains motif starting time and recordings
    song_length: length of motif
    bin_size: how many datapoints to combine into a bin
    num_clusters: how many clusters of neurons are recorded
    start_extend_bins: at the beginning of the motif, how many bins do you want to shift. Shifting the start to the left
                        is negative, and shifting to the right is positive. Default to be 0
    end_extend_bins: same as start_extend_bins, how many bins to shift at the end of motif. Shifting left is negative and
                        right is positive. Default to be 0.
    
    '''
    bird_folder_data, bird_folder_save, repos_folder, results_folder = locate_folders(bird_id, model_name)
    save_name = 'mel_%02d_%d_%d_%s.p' %(num_lookbacks, start_extend_bins, end_extend_bins, specify_subdir)
    
    fft_size = bin_size*16 # window size for the FFT
    step_size = bin_size # distance to slide along the window (in time)
    start_extend_bins *= int(fft_size/bin_size)
    end_extend_bins *= int(fft_size/bin_size)
    spec_thresh = 5 # threshold for spectrograms (lower filters out more noise)
    lowcut = 500 # Hz # Low cut for our butter bandpass filter
    highcut = 15000 # Hz # High cut for our butter bandpass filter
    # For mels
    # number of mel frequency channels
    shorten_factor = 1 # how much should we compress the x-axis (time)
     # Hz # What frequency to start sampling our melS from 
     # Hz # What frequency to stop sampling our melS from 
    
    mel_filter, mel_inversion_filter = create_mel_filter(fft_size = fft_size,
                                                        n_freq_components = n_mel_freq_components,
                                                        start_freq = start_freq,
                                                        end_freq = end_freq)
    
    num_bins = int(song_length/bin_size-2-start_extend_bins+end_extend_bins)#number of bins in each song
    
    if specify_subdir:
        subdirs = [os.path.join(bird_folder_data, specify_subdir)]
    else:
        subdirs = [combined[0] for combined in os.walk(bird_folder_data)][1:]
        
    extended_song_length = song_length+bin_size*(end_extend_bins-start_extend_bins)
    input_song_length = extended_song_length+(num_lookbacks-2)*bin_size 
    #take out two bins due to mel processing algorithm
    
    input_target_comb = list()
    tot_motifs = 0
    
    for directory in subdirs:
        #kwik_file = os.path.join(directory, neural_file_name)
        song_file = os.path.join(directory, song_file_name)
        
        try:
            trial1 = h5py.File(song_file,'r')['event_types/singing/motiff_1/time_samples'].value
        except IOError:
            print('--->Skipping '+directory+' because it doesnt contain spiking kwik files.')
            continue
            
        if "sleep" in directory:
            print('--->Skipping '+directory+' because it is a sleep recording.')
            continue
            
        recording = directory[directory.index('day-'):]
        recording_folder_save = os.path.join(bird_folder_save, recording)
        if not os.path.exists(recording_folder_save):
            os.makedirs(recording_folder_save)
            
        print('Start working on '+directory)
        
        '''
        with h5py.File(kwik_file,'r') as f:
            time_samples = f['/channel_groups/0/spikes/time_samples'].value
            cluster_num = f['/channel_groups/0/spikes/clusters/main'].value
            recording_num = f['/channel_groups/0/spikes/recording'].value
            if max(recording_num)>20:
                error_indices = [i for i,v in enumerate(list(recording_num)) if v > 20]
                for index in error_indices:
                    recording_num[index] = recording_num[index-1]
                print('***Erroneous indices fixed.')
                
            #print(type(time_samples))
            rec_counter = collections.Counter(recording_num)
            rec_list = rec_counter.keys()
            spike_counts = rec_counter.values()
            time_samples_each_rec = list()
            cluster_each_rec = list()

            #separate time samples based on recording
            for rec in rec_list:
                #start_samples.append(f['/recordings/'][str(rec)].attrs['start_sample'])
                prev_num_samples = sum(spike_counts[:rec])
                time_samples_each_rec.append(time_samples[prev_num_samples:prev_num_samples+spike_counts[rec]])
                #time_samples[prev_num_samples:prev_num_samples+spike_counts[rec]] = time_samples[
                #    prev_num_samples:prev_num_samples+spike_counts[rec]]+f['/recordings/'][str(rec)].attrs['start_sample']
                cluster_each_rec.append(cluster_num[prev_num_samples:prev_num_samples+spike_counts[rec]])

            #print(np.shape(time_samples_each_rec))
            #print(time_samples_each_rec)
            '''
            
        with h5py.File(song_file, 'r') as f:
            motif_starts = f['event_types/singing/motiff_1/time_samples'].value  #start time of each motif
            motif_recs = f['event_types/singing/motiff_1/recording'].value  #recording number of each motif
            songtuples = zip(motif_recs, motif_starts)
            
            tot_motifs += len(songtuples)

            for rec, ideal_start in songtuples:
                
                song_ideal_start = ideal_start+bin_size*start_extend_bins
                song_ideal_end = song_ideal_start+extended_song_length
                
                input_ideal_start = song_ideal_start-num_lookbacks*bin_size
                
                ## 09/29/17 copy something from here
                input_ideal_end = song_ideal_end+int((16-num_lookbacks)%16)*bin_size
                
                #process wav file into mel spec
                wav_name = 'experiment-rec_0%02d.mic.wav' % (rec)
                wav_file = os.path.join(directory, wav_name)
                
                wav_rate, wav_data = wavfile.read(wav_file)
                wav_data = wav_data[input_ideal_start:input_ideal_end]
                wav_data = butter_bandpass_filter(wav_data, lowcut, highcut, wav_rate, order=1)
                #print(len(wav_data_input), len(wav_data_target))
                #if len(wav_data)!=extended_song_length:
                #    raise ValueError('less wav datapointes loaded than expected')
                
                #plot each filtered unmel-edwav
                #fig = plt.figure()
                #plt.plot(wav_data)
                #plt.title(str(rec)+': '+str(song_ideal_start))
                #plt.savefig(os.path.join(recording_folder_save, 'motif '+str(rec)+' '+str(ideal_start)+'.png'))
                #plt.close(fig)
                
                #mel and compile into a list based on time bins
                wav_spectrogram = pretty_spectrogram(wav_data.astype('float64'), fft_size = fft_size, 
                                   step_size = step_size, log = True, thresh = spec_thresh)
                #print(len(list(wav_spectrogram_input)))
                mel_list = list(make_mel(wav_spectrogram, mel_filter, shorten_factor = shorten_factor).transpose())
                
                #print(len(mel_list_target))
                '''
                current_t_list = time_samples_each_rec[rec] #extract neural activities for current motif
                current_cluster_list = cluster_each_rec[rec]  #extract cluster info for current motif

                neural_start_index = bisect.bisect_left(current_t_list, neural_ideal_start) #locate the index of starting point of the current motif
                neural_start_val = current_t_list[neural_start_index] #the actual starting point of the current motif
                #start_empty_bins = (neural_start_val-neural_ideal_start)//bin_size
                
                neural_end_index = bisect.bisect_left(current_t_list, neural_ideal_end)-1 #the end index of the current motif wrt indexing within the entire recording
                #end_empty_bins = (neural_ideal_end-current_t_list[neural_end_index])//bin_size

                t_during_song = current_t_list[neural_start_index:neural_end_index+1]-neural_ideal_start#normalize time sequence
                cluster_during_song = current_cluster_list[neural_start_index:neural_end_index+1] #extract cluster sequence

                counts_list = list() #contains all counts data for this motif
                for i in range(num_bins+num_lookbacks):
                    bin_start_index = bisect.bisect_left(t_during_song, i*bin_size)
                    bin_end_index = bisect.bisect_left(t_during_song,(i+1)*bin_size)
                    counts, bins = np.histogram(cluster_during_song[bin_start_index:bin_end_index], bins=np.arange(0,num_clusters+1))
                    counts_list.append(counts)
                    '''
                    
                for i in range(num_bins):
                    input_target_tuple = (mel_list[i:i+num_lookbacks], mel_list[i+num_lookbacks])
                    input_target_comb.append(input_target_tuple)

                """for cluster in range(num_clusters):
                    index_this_cluster = [i for i, j in enumerate(cluster_during_song) if j == cluster]
                    t_this_cluster = t_during_song[index_this_cluster]
                    counts,bins = np.histogram(t_this_cluster, bins=[start_val:bin_size:end_val, sys.maxint])
                    """

                #if counts_list[0].shape[0] != num_clusters:
                #    raise ValueError('check cluster histogram')
                    
                #if len(counts_list)-num_lookbacks != len(mel_list):
                #    raise ValueError('Neural spiking bins do not correspond to spectrogram bins.')

                    #check for counting errors
        print('Finished.')
        
    if len(input_target_comb)!= num_bins*tot_motifs:
        raise ValueError('Might be missing some motifs or bins')
        
    pickle.dump(input_target_comb, open(os.path.join(bird_folder_save, save_name),'wb'))
        
    return input_target_comb


# ### Zeke's Parameter Prediction

# In[ ]:


def make_datasets_finch_alphabeta(neural_file_name, parameter_file_name, song_length, bin_size, num_clusters, num_lookbacks, 
                                bird_id, model_name, parameter,
                                start_extend_bins=0, end_extend_bins=0, specify_subdir=None,
                                 fixed_length = True, blanks_ratio = 0, shuffle = False):
    '''
    neural_file_name: kwik file that contains neural data
    song_file_name: kwik file that contains motif starting time and recordings
    song_length: length of motif
    bin_size: how many datapoints to combine into a bin
    num_clusters: how many clusters of neurons are recorded
    start_extend_bins: at the beginning of the motif, how many bins do you want to shift. Shifting the start to the left
                        is negative, and shifting to the right is positive. Default to be 0
    end_extend_bins: same as start_extend_bins, how many bins to shift at the end of motif. Shifting left is negative and
                        right is positive. Default to be 0.
    peak_var: variance of acceptable peaks. the lowest peak value to be considered peak is max(peaks)*peak_var
    noise_thresh: threshold to be considered as noise
    fixed_length: whether or not to extend the song when syllables are cut off at the orignal start/end point
    to_cat: output to category. If true, output will be [0, 1] or [1, 0] instead of [0] or [1]
    mode: 'fft' uses fft syllable detection, 'wav' uses wav syllable detection
    blanks_ratio: whether to add blanks alongside the detected motifs
    
    '''
    if parameter not in ['alpha', 'beta']:
        raise ValueError('Parameter can only be alpha or beta.')
    
    if parameter=='alpha':
        parameter = 'env'
        
    bird_folder_data, bird_folder_save, repos_folder, results_folder = locate_folders(bird_id, model_name)
    save_name = 'par_%02d_%d_%d.p' %(num_lookbacks, start_extend_bins, end_extend_bins)
    
    subdirs = which_dir(bird_folder_data, specify_subdir)

    par_neural_comb=list()
    
    for directory in subdirs:
        
        recording = directory[directory.index('day-'):]
        recording_folder_save = os.path.join(bird_folder_save, recording)
        
        kwik_file = os.path.join(directory, neural_file_name)
        
        
        try:
            trial1 = h5py.File(kwik_file,'r')['/channel_groups/0/spikes/time_samples'].value
        except IOError:
            print('--->Skipping '+directory+' because it doesnt contain spiking kwik files.')
            continue
            
        if "sleep" in directory:
            print('--->Skipping '+directory+' because it is a sleep recording.')
            continue
        
        if not os.path.exists(recording_folder_save):
            os.makedirs(recording_folder_save)
        
        time_samples_each_rec, cluster_each_rec = separate_recs(directory, neural_file_name, bird_id, model_name)
            
        par_neural_comb_dir = calculate_neural_alphabeta(directory, time_samples_each_rec, cluster_each_rec, 
                                                     recording_folder_save, parameter,
                                                     neural_file_name, parameter_file_name, bird_id, model_name, num_clusters, bin_size, 
                                                     song_length, num_lookbacks, start_extend_bins, 
                                                     end_extend_bins, fixed_length = fixed_length, 
                                                        blanks_ratio = blanks_ratio)
        
        par_neural_comb += par_neural_comb_dir
        #mel_neural_comb is a collection of lists, each inidividual list is a song, contains a colletion of neural-mel 
        #tuples within a song
        
        print('Finished.')
    
    if shuffle:
        random.shuffle(par_neural_comb)
    par_neural_comb = song_based_to_all(par_neural_comb)
        
    #regoup into batch based   
    pickle.dump(par_neural_comb, open(os.path.join(bird_folder_save, save_name),'wb'))
        
    return par_neural_comb


# In[ ]:


def calculate_neural_alphabeta(directory, time_samples_each_rec, cluster_each_rec,  
                         recording_folder_save, parameter,
                         neural_file_name, parameter_file_name, bird_id, model_name, num_clusters, bin_size, 
                         song_length, num_lookbacks, start_extend_bins, 
                         end_extend_bins, fixed_length = True, blanks_ratio = 0):
    
    start_extend_pts = bin_size*start_extend_bins
    end_extend_pts = bin_size*end_extend_bins
    extended_song_length = song_length+start_extend_pts+end_extend_pts
    
    par_file = os.path.join(recording_folder_save, parameter_file_name) 
        
    par_neural_comb = list()
    
    fig_folder_save = os.path.join(recording_folder_save, 'alphabeta', '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now()))
    
    if not os.path.exists(fig_folder_save):
        os.makedirs(fig_folder_save)
    
    count = 0
    
    all_fit_syl = pickle.load( open( par_file, "rb" ) )
    
    songtuples = sorted(list(set(zip(all_fit_syl.rec, all_fit_syl.mot_start))))
    
    if blanks_ratio:
        songtuples += generate_blanks(songtuples, song_length, ratio = blanks_ratio)

    for rec, ideal_start in songtuples:

        par_neural_comb_song = list()

        song_start = ideal_start-start_extend_pts
        song_end = song_start+extended_song_length
        
        if ideal_start in list(all_fit_syl.mot_start):
            par = np.concatenate(all_fit_syl[all_fit_syl.mot_start==ideal_start][parameter].tolist())
        else:
            par = [0]*int(extended_song_length/bin_size)

        fig1 = plt.figure()
        plt.plot(par)
        plt.title(str(rec)+': '+str(song_start))
        plt.savefig(os.path.join(fig_folder_save, '%03d.png' %(count)))
        plt.close(fig1)

        #print(heads)
        song_actual_start = song_start
        song_actual_end = song_start+extended_song_length
        num_bins = int(extended_song_length/bin_size)
            #print(num_bins)

        if int(song_actual_end-song_actual_start)!=int(num_bins*bin_size):
            raise ValueError('Check num of bins calculation')

        neural_actual_start = song_actual_start-num_lookbacks*bin_size
        neural_actual_end = song_actual_end

        current_t_list = time_samples_each_rec[rec] #extract neural activities for current motif
        current_cluster_list = cluster_each_rec[rec]  #extract cluster info for current motif

        counts_list = spike_counter(current_t_list, current_cluster_list, neural_actual_start, neural_actual_end, song_actual_start, rec,
                                      recording_folder_save,
                                      num_clusters, bin_size, num_bins, num_lookbacks, plot_raster=False)

        for i in range(num_bins):
            inputs = np.array(counts_list[i:i+num_lookbacks])
            output = par[i]
            par_neural_tuple = (inputs, output)
            par_neural_comb_song.append(par_neural_tuple)

        par_neural_comb.append(par_neural_comb_song)
        count += 1
        
    return par_neural_comb


# ### Syllable Index Prediction

# In[ ]:


def make_datasets_finch_syllable(neural_file_name, song_file_name, song_length, bin_size, num_clusters, num_lookbacks, 
                                 fft_size, bird_id, model_name,
                                start_extend_bins=0, end_extend_bins=0, specify_subdir=None,
                                n_mel_freq_components = 64, start_freq = 200, end_freq = 15000, peak_var = 0.6, noise_thresh=47, 
                                 fixed_length = False, to_cat = False, mode = 'wav', blanks_ratio = 0, shuffle = False):
    '''
    neural_file_name: kwik file that contains neural data
    song_file_name: kwik file that contains motif starting time and recordings
    song_length: length of motif
    bin_size: how many datapoints to combine into a bin
    num_clusters: how many clusters of neurons are recorded
    start_extend_bins: at the beginning of the motif, how many bins do you want to shift. Shifting the start to the left
                        is negative, and shifting to the right is positive. Default to be 0
    end_extend_bins: same as start_extend_bins, how many bins to shift at the end of motif. Shifting left is negative and
                        right is positive. Default to be 0.
    peak_var: variance of acceptable peaks. the lowest peak value to be considered peak is max(peaks)*peak_var
    noise_thresh: threshold to be considered as noise
    fixed_length: whether or not to extend the song when syllables are cut off at the orignal start/end point
    to_cat: output to category. If true, output will be [0, 1] or [1, 0] instead of [0] or [1]
    mode: 'fft' uses fft syllable detection, 'wav' uses wav syllable detection
    blanks_ratio: whether to add blanks alongside the detected motifs
    
    '''
    
    if blanks_ratio:
        fixed_length = True
        
    bird_folder_data, bird_folder_save, repos_folder, results_folder = locate_folders(bird_id, model_name)
    save_name = 'syl_%02d_%d_%d.p' %(num_lookbacks, start_extend_bins, end_extend_bins)
    
    subdirs = which_dir(bird_folder_data, specify_subdir)
    
    '''
    all_batches = list()
    for _ in range(num_bins):
        all_batches.append(list())
    '''
    syl_neural_comb=list()
    
    for directory in subdirs:
        kwik_file = os.path.join(directory, neural_file_name)
        song_file = os.path.join(directory, song_file_name)
        
        try:
            trial1 = h5py.File(kwik_file,'r')['/channel_groups/0/spikes/time_samples'].value
        except IOError:
            print('--->Skipping '+directory+' because it doesnt contain spiking kwik files.')
            continue
            
        if "sleep" in directory:
            print('--->Skipping '+directory+' because it is a sleep recording.')
            continue
        
        recording = directory[directory.index('day-'):]
        recording_folder_save = os.path.join(bird_folder_save, recording)
        if not os.path.exists(recording_folder_save):
            os.makedirs(recording_folder_save)
        
        time_samples_each_rec, cluster_each_rec = separate_recs(directory, neural_file_name, bird_id, model_name)
            
        syl_neural_comb_dir = calculate_neural_syllable(directory, time_samples_each_rec, cluster_each_rec, 
                                                     recording_folder_save, 
                                                     neural_file_name, song_file_name, bird_id, model_name, num_clusters, bin_size, 
                                                     song_length, num_lookbacks, peak_var, noise_thresh, start_freq, end_freq, start_extend_bins, 
                                                     end_extend_bins, fixed_length = fixed_length, fft_size = fft_size,
                                                      to_cat = to_cat, mode = mode, blanks_ratio = blanks_ratio)
        
        syl_neural_comb += syl_neural_comb_dir
        #mel_neural_comb is a collection of lists, each inidividual list is a song, contains a colletion of neural-mel 
        #tuples within a song
        
        print('Finished.')
    
    if shuffle:
        random.shuffle(syl_neural_comb)
    syl_neural_comb = song_based_to_all(syl_neural_comb)
        
    #regoup into batch based   
    pickle.dump(syl_neural_comb, open(os.path.join(bird_folder_save, save_name),'wb'))
        
    return syl_neural_comb


# In[ ]:


def calculate_neural_syllable(directory, time_samples_each_rec, cluster_each_rec,  
                         recording_folder_save,
                         neural_file_name, song_file_name, bird_id, model_name, num_clusters, bin_size, 
                         song_length, num_lookbacks, peak_var, noise_thresh, start_freq, end_freq, start_extend_bins, 
                         end_extend_bins, labels=None, flatten_inputs = False, fixed_length = False, fft_size = 256, 
                              to_cat = False, mode = 'wav', blanks_ratio = 0):
    
    start_extend_pts = bin_size*start_extend_bins
    end_extend_pts = bin_size*end_extend_bins
    extended_song_length = song_length+start_extend_pts+end_extend_pts
    
    if song_file_name.endswith('kwik') or song_file_name.endswith('kwe'):
        song_file_type = 'kwik'
    else:
        song_file_type = 'pickle'
    song_file = os.path.join(directory, song_file_name)
        
    syl_neural_comb = list()
    
    fig_folder_save = os.path.join(recording_folder_save, 'syllable', '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now()))
    
    if not os.path.exists(fig_folder_save):
        os.makedirs(fig_folder_save)
    
    count = 0
    
    if song_file_type=='kwik':
        motif_starts = h5py.File(song_file, 'r')['event_types/singing/motiff_1/time_samples'].value  #start time of each motif
        motif_recs = h5py.File(song_file, 'r')['event_types/singing/motiff_1/recording'].value  #recording number of each motif
        songtuples = zip(motif_recs, motif_starts)
        last_rec = -1
        if blanks_ratio:
            songtuples += generate_blanks(songtuples, song_length, ratio = blanks_ratio)
            
        for rec, ideal_start in songtuples:
            
            syl_neural_comb_song = list()

            song_start = ideal_start-start_extend_pts
            song_end = song_start+extended_song_length
                
            #process wav file into mel spec
            
            step_size = bin_size # distance to slide along the window (in time)
            spec_thresh = 5 # threshold for spectrograms (lower filters out more noise)
            lowcut = 500 # Hz # Low cut for our butter bandpass filter
            highcut = 14999 # Hz # High cut for our butter bandpass filter
            # For mels
            shorten_factor = 1 # how much should we compress the x-axis (time)
            
            if last_rec != rec:
                wav_name = 'experiment-rec_0%02d.mic.wav' % (rec)
                wav_file = os.path.join(directory, wav_name)
                wav_rate, wav_data_rec = wavfile.read(wav_file)
                wav_data_rec = butter_bandpass_filter(wav_data_rec, lowcut, highcut, wav_rate, order=1)
                last_rec = rec
            
            fig1 = plt.figure()
            plt.plot(wav_data_rec[song_start:song_end])
            plt.title(str(rec)+': '+str(song_start))
            plt.savefig(os.path.join(fig_folder_save, '%03d_original.png' %(count)))
            plt.close(fig1)
            
            heads, tails, lengths = find_syllable_boundaries(wav_data_rec, song_start, song_end, wav_rate, fft_size, step_size, count, spec_thresh=spec_thresh, 
                                     peak_var = peak_var, noise_thresh=noise_thresh, fig_folder_save = fig_folder_save, mode=mode)
            
            #print(heads)
            if not blanks_ratio:
                heads_wav = bins2wav(heads, step_size)
                tails_wav = bins2wav(tails, step_size)
                lengths_wav = bins2wav(lengths, step_size)

                offset_bins = heads[0]
            
            if fixed_length:
                song_actual_start = song_start
                song_actual_end = song_start+extended_song_length
                num_bins = int(extended_song_length/step_size)
                #print(num_bins)
            else:
                num_bins = int(tails[-1]-heads[0])
                song_actual_start = song_start+heads_wav[0]
                song_actual_end = song_start+tails_wav[-1]
                heads = [head-offset_bins for head in heads]
                if heads[0] != 0:
                    raise ValueError('Bins incorrectly initialized')
                tails = [tail-offset_bins for tail in tails]
            
            if int(song_actual_end-song_actual_start)!=int(num_bins*step_size):
                raise ValueError('Check num of bins calculation')
                
            #code syllable index
            syllable = np.array([0]*num_bins)
                
            if len(heads)==5:
                for i in range(5):
                    syllable[heads[i]:tails[i]] = i+1
            elif len(heads):
                continue
                
            wav_data = wav_data_rec[song_actual_start:song_actual_end]
            
            neural_actual_start = song_actual_start-num_lookbacks*bin_size
            neural_actual_end = song_actual_end
            

            #plot each filtered unmel-edwav
            fig2 = plt.figure()
            plt.plot(syllable)
            plt.title(str(rec)+': '+str(song_start))
            plt.savefig(os.path.join(fig_folder_save, '%03d.png' %(count)))
            plt.close(fig2)
            
            #create 1/0 classifiers
            
            current_t_list = time_samples_each_rec[rec] #extract neural activities for current motif
            current_cluster_list = cluster_each_rec[rec]  #extract cluster info for current motif

            counts_list = spike_counter(current_t_list, current_cluster_list, neural_actual_start, neural_actual_end, song_actual_start, rec,
                                          recording_folder_save,
                                          num_clusters, bin_size, num_bins, num_lookbacks, plot_raster=False)
                
            syl_neural_comb_song = list()
            
            for i in range(num_bins):
                inputs = np.array(counts_list[i:i+num_lookbacks])
                outputs = syllable[i]
                if to_cat:
                    outputs = np.array([0]*6)
                    outputs[int(syllable[i])] = 1
                syl_neural_tuple = (inputs, outputs)
                syl_neural_comb_song.append(syl_neural_tuple)
                
            syl_neural_comb.append(syl_neural_comb_song)
            count += 1
    '''       
    else:
        try:
            pickle_data = pd.read_pickle(song_file)
        except:
            song_file = os.path.join('/mnt/cube/earneodo/bci_zf/proc_data/', bird_id, song_file_name)
        pickle_data = pd.read_pickle(song_file)
        indeces = [i for i, j in enumerate(pickle_data['recon_folder']) if directory in j]
        indeces = [i for i, j in zip(indeces, pickle_data['syllable_labels'][indeces]) if j in labels]
        total_bins = 0
            
        for ind in indeces:
            mel_neural_comb_song = list()
            wav_file = pickle_data['recon_folder'][ind]
            rec = int(wav_file[wav_file.index('rec_')+4:wav_file.index('rec_')+7])
            ori_song_length = int(pickle_data['recon_length'][ind]*30000)
            song_length, neural_length = cal_modified_length(ori_song_length, bin_size, start_extend_bins=0, end_extend_bins=0, 
                                                                 num_lookbacks=num_lookbacks)
            num_bins = int(np.ceil(ori_song_length/bin_size-2))

            total_bins+=num_bins

            song_ideal_start = int(pickle_data['recon_t_rel_wav'][ind]*30000)
            song_ideal_end = song_ideal_start+song_length
            song_ideal_end = song_ideal_end+int((16-num_lookbacks)%16)*bin_size

            neural_ideal_start = song_ideal_start+bin_size*(1-num_lookbacks)
            neural_ideal_end = neural_ideal_start+neural_song_length

            mel_list = make_specs(wav_file, rec, song_ideal_start, song_ideal_end, recording_folder_save, bin_size, n_mel_freq_components, 
                                      start_freq, end_freq)

            current_t_list = time_samples_each_rec[rec] #extract neural activities for current motif
            current_cluster_list = cluster_each_rec[rec]  #extract cluster info for current motif

            counts_list = spike_counter(current_t_list, current_cluster_list, neural_ideal_start, neural_ideal_end, song_ideal_start, rec,
                                          recording_folder_save,
                                          num_clusters, bin_size, num_bins, num_lookbacks)

            for i in range(num_bins):
                mel_neural_tuple = (np.array(counts_list[i:i+num_lookbacks]), np.array(mel_list[i]))
                mel_neural_comb_song.append(mel_neural_tuple)
            
            mel_neural_comb.append(mel_neural_comb_song)
    '''            
    return syl_neural_comb


# ### Timecode Prediction

# In[ ]:


def make_datasets_finch_timecode(neural_file_name, song_file_name, song_length, bin_size, num_clusters, num_lookbacks, 
                                 fft_size, bird_id, model_name,
                                start_extend_bins=0, end_extend_bins=0, specify_subdir=None,
                                n_mel_freq_components = 64, start_freq = 200, end_freq = 15000, peak_var = 0.6, noise_thresh=47, 
                                 fixed_length = False, to_cat = False, mode = 'fft', blanks_ratio = 0, shuffle = False, time_seq=False):
    '''
    neural_file_name: kwik file that contains neural data
    song_file_name: kwik file that contains motif starting time and recordings
    song_length: length of motif
    bin_size: how many datapoints to combine into a bin
    num_clusters: how many clusters of neurons are recorded
    start_extend_bins: at the beginning of the motif, how many bins do you want to shift. Shifting the start to the left
                        is negative, and shifting to the right is positive. Default to be 0
    end_extend_bins: same as start_extend_bins, how many bins to shift at the end of motif. Shifting left is negative and
                        right is positive. Default to be 0.
    peak_var: variance of acceptable peaks. the lowest peak value to be considered peak is max(peaks)*peak_var
    noise_thresh: threshold to be considered as noise
    fixed_length: whether or not to extend the song when syllables are cut off at the orignal start/end point
    to_cat: output to category. If true, output will be [0, 1] or [1, 0] instead of [0] or [1]
    mode: 'fft' uses fft syllable detection, 'wav' uses wav syllable detection
    blanks_ratio: whether to add blanks alongside the detected motifs
    
    '''
    
    if blanks_ratio:
        fixed_length = True
        
    bird_folder_data, bird_folder_save, repos_folder, results_folder = locate_folders(bird_id, model_name)
    save_name = 'tc_%02d_%d_%d.p' %(num_lookbacks, start_extend_bins, end_extend_bins)
    
    subdirs = which_dir(bird_folder_data, specify_subdir)
    
    '''
    all_batches = list()
    for _ in range(num_bins):
        all_batches.append(list())
    '''
    tc_neural_comb=list()
    
    for directory in subdirs:
        kwik_file = os.path.join(directory, neural_file_name)
        song_file = os.path.join(directory, song_file_name)
        
        try:
            trial1 = h5py.File(kwik_file,'r')['/channel_groups/0/spikes/time_samples'].value
        except IOError:
            print('--->Skipping '+directory+' because it doesnt contain spiking kwik files.')
            continue
            
        if "sleep" in directory:
            print('--->Skipping '+directory+' because it is a sleep recording.')
            continue
        
        recording = directory[directory.index('day-'):]
        recording_folder_save = os.path.join(bird_folder_save, recording)
        if not os.path.exists(recording_folder_save):
            os.makedirs(recording_folder_save)
        
        time_samples_each_rec, cluster_each_rec = separate_recs(directory, neural_file_name, bird_id, model_name)
            
        tc_neural_comb_dir = calculate_neural_timecode(directory, time_samples_each_rec, cluster_each_rec, 
                                                     recording_folder_save, 
                                                     neural_file_name, song_file_name, bird_id, model_name, num_clusters, bin_size, 
                                                     song_length, num_lookbacks, peak_var, noise_thresh, start_freq, end_freq, start_extend_bins, 
                                                     end_extend_bins, fixed_length = fixed_length, fft_size = fft_size,
                                                      to_cat = to_cat, mode = mode, blanks_ratio = blanks_ratio, time_seq= time_seq)
        
        tc_neural_comb += tc_neural_comb_dir
        #mel_neural_comb is a collection of lists, each inidividual list is a song, contains a colletion of neural-mel 
        #tuples within a song
        
        print('Finished.')
    
    if shuffle:
        random.shuffle(tc_neural_comb)
    tc_neural_comb = song_based_to_all(tc_neural_comb)
        
    #regoup into batch based   
    pickle.dump(tc_neural_comb, open(os.path.join(bird_folder_save, save_name),'wb'))
        
    return tc_neural_comb


# In[ ]:


def calculate_neural_timecode(directory, time_samples_each_rec, cluster_each_rec,  
                         recording_folder_save,
                         neural_file_name, song_file_name, bird_id, model_name, num_clusters, bin_size, 
                         song_length, num_lookbacks, peak_var, noise_thresh, start_freq, end_freq, start_extend_bins, 
                         end_extend_bins, labels=None, flatten_inputs = False, fixed_length = False, fft_size = 256, 
                              to_cat = False, mode = 'fft', blanks_ratio = 0, time_seq=False):
    
    start_extend_pts = bin_size*start_extend_bins
    end_extend_pts = bin_size*end_extend_bins
    extended_song_length = song_length+start_extend_pts+end_extend_pts
    
    if song_file_name.endswith('kwik') or song_file_name.endswith('kwe'):
        song_file_type = 'kwik'
    else:
        song_file_type = 'pickle'
    song_file = os.path.join(directory, song_file_name)
        
    tc_neural_comb = list()
    
    fig_folder_save = os.path.join(recording_folder_save, 'timecode', '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now()))
    
    if not os.path.exists(fig_folder_save):
        os.makedirs(fig_folder_save)
    
    count = 0
    
    if song_file_type=='kwik':
        motif_starts = h5py.File(song_file, 'r')['event_types/singing/motiff_1/time_samples'].value  #start time of each motif
        motif_recs = h5py.File(song_file, 'r')['event_types/singing/motiff_1/recording'].value  #recording number of each motif
        songtuples = zip(motif_recs, motif_starts)
        last_rec = -1
        if blanks_ratio:
            songtuples += generate_blanks(songtuples, song_length, ratio = blanks_ratio)
            
        for rec, ideal_start in songtuples:
            
            tc_neural_comb_song = list()

            song_start = ideal_start-start_extend_pts
            song_end = song_start+extended_song_length
                
            #process wav file into mel spec
            
            step_size = bin_size # distance to slide along the window (in time)
            spec_thresh = 5 # threshold for spectrograms (lower filters out more noise)
            lowcut = 500 # Hz # Low cut for our butter bandpass filter
            highcut = 14999 # Hz # High cut for our butter bandpass filter
            # For mels
            shorten_factor = 1 # how much should we compress the x-axis (time)
            
            if last_rec != rec:
                wav_name = 'experiment-rec_0%02d.mic.wav' % (rec)
                wav_file = os.path.join(directory, wav_name)
                wav_rate, wav_data_rec = wavfile.read(wav_file)
                wav_data_rec = butter_bandpass_filter(wav_data_rec, lowcut, highcut, wav_rate, order=1)
                last_rec = rec
            
            fig1 = plt.figure()
            plt.plot(wav_data_rec[song_start:song_end])
            plt.title(str(rec)+': '+str(song_start))
            plt.savefig(os.path.join(fig_folder_save, '%03d_original.png' %(count)))
            plt.close(fig1)
            
            heads, tails, lengths = find_syllable_boundaries(wav_data_rec, song_start, song_end, wav_rate, fft_size, step_size, count, spec_thresh=spec_thresh, 
                                     peak_var = peak_var, noise_thresh=noise_thresh, fig_folder_save = fig_folder_save, mode=mode)
            
            #print(heads)
            if not blanks_ratio:
                heads_wav = bins2wav(heads, step_size)
                tails_wav = bins2wav(tails, step_size)
                lengths_wav = bins2wav(lengths, step_size)

                offset_bins = heads[0]
            
            if fixed_length:
                song_actual_start = song_start
                song_actual_end = song_start+extended_song_length
                num_bins = int(extended_song_length/step_size)
                #print(num_bins)
            else:
                num_bins = int(tails[-1]-heads[0])
                song_actual_start = song_start+heads_wav[0]
                song_actual_end = song_start+tails_wav[-1]
                heads = [head-offset_bins for head in heads]
                if heads[0] != 0:
                    raise ValueError('Bins incorrectly initialized')
                tails = [tail-offset_bins for tail in tails]
            
            if int(song_actual_end-song_actual_start)!=int(num_bins*step_size):
                raise ValueError('Check num of bins calculation')
                
            durations = list()
            
            if heads:
                for head, tail in zip(heads, tails):
                    durations+=range(head, tail)
            else:
                durations = []
            
            wav_data = wav_data_rec[song_actual_start:song_actual_end]
            on_off = [(i in durations) for i in range(num_bins)]
            
            neural_actual_start = song_actual_start-num_lookbacks*bin_size
            neural_actual_end = song_actual_end
            

            #plot each filtered unmel-edwav
            fig2 = plt.figure()
            plt.plot(wav_data)
            plt.title(str(rec)+': '+str(song_start))
            plt.savefig(os.path.join(fig_folder_save, '%03d.png' %(count)))
            plt.close(fig2)
            
            #create 1/0 classifiers
            
            current_t_list = time_samples_each_rec[rec] #extract neural activities for current motif
            current_cluster_list = cluster_each_rec[rec]  #extract cluster info for current motif

            counts_list = spike_counter(current_t_list, current_cluster_list, neural_actual_start, neural_actual_end, song_actual_start, rec,
                                          recording_folder_save,
                                          num_clusters, bin_size, num_bins, num_lookbacks, plot_raster=False)
                
            tc_neural_comb_song = list()
            
            for i in range(num_bins):
                inputs = np.array(counts_list[i:i+num_lookbacks])
                if time_seq:
                    tc = [i]
                elif to_cat:
                    if i in durations:
                        tc = [1, 0]
                    else:
                        tc = [0, 1]
                else:
                    if i in durations:
                        tc = [1]
                    else:
                        tc = [0]
                tc_neural_tuple = (inputs, tc)
                tc_neural_comb_song.append(tc_neural_tuple)
                
            tc_neural_comb.append(tc_neural_comb_song)
            count += 1
    '''       
    else:
        try:
            pickle_data = pd.read_pickle(song_file)
        except:
            song_file = os.path.join('/mnt/cube/earneodo/bci_zf/proc_data/', bird_id, song_file_name)
        pickle_data = pd.read_pickle(song_file)
        indeces = [i for i, j in enumerate(pickle_data['recon_folder']) if directory in j]
        indeces = [i for i, j in zip(indeces, pickle_data['syllable_labels'][indeces]) if j in labels]
        total_bins = 0
            
        for ind in indeces:
            mel_neural_comb_song = list()
            wav_file = pickle_data['recon_folder'][ind]
            rec = int(wav_file[wav_file.index('rec_')+4:wav_file.index('rec_')+7])
            ori_song_length = int(pickle_data['recon_length'][ind]*30000)
            song_length, neural_length = cal_modified_length(ori_song_length, bin_size, start_extend_bins=0, end_extend_bins=0, 
                                                                 num_lookbacks=num_lookbacks)
            num_bins = int(np.ceil(ori_song_length/bin_size-2))

            total_bins+=num_bins

            song_ideal_start = int(pickle_data['recon_t_rel_wav'][ind]*30000)
            song_ideal_end = song_ideal_start+song_length
            song_ideal_end = song_ideal_end+int((16-num_lookbacks)%16)*bin_size

            neural_ideal_start = song_ideal_start+bin_size*(1-num_lookbacks)
            neural_ideal_end = neural_ideal_start+neural_song_length

            mel_list = make_specs(wav_file, rec, song_ideal_start, song_ideal_end, recording_folder_save, bin_size, n_mel_freq_components, 
                                      start_freq, end_freq)

            current_t_list = time_samples_each_rec[rec] #extract neural activities for current motif
            current_cluster_list = cluster_each_rec[rec]  #extract cluster info for current motif

            counts_list = spike_counter(current_t_list, current_cluster_list, neural_ideal_start, neural_ideal_end, song_ideal_start, rec,
                                          recording_folder_save,
                                          num_clusters, bin_size, num_bins, num_lookbacks)

            for i in range(num_bins):
                mel_neural_tuple = (np.array(counts_list[i:i+num_lookbacks]), np.array(mel_list[i]))
                mel_neural_comb_song.append(mel_neural_tuple)
            
            mel_neural_comb.append(mel_neural_comb_song)
    '''            
    return tc_neural_comb


# ### Starling Spec Prediction

# In[ ]:


def make_datasets_starling(neural_file_name, bird_song_file, specify_dir, bin_size, num_clusters, num_lookbacks, labels,
                           bird_id, model_name,
                            n_mel_freq_components = 64, start_freq = 200, end_freq = 15000):
    '''
    This functionw works on starling data where song info is stored in pickle file.
    neural_file_name: kwik file that contains neural data
    song_file_name: kwik file that contains motif starting time and recordings
    song_length: length of motif
    bin_size: how many datapoints to combine into a bin
    num_clusters: how many clusters of neurons are recorded
    start_extend_bins: at the beginning of the motif, how many bins do you want to shift. Shifting the start to the left
                        is negative, and shifting to the right is positive. Default to be 0
    end_extend_bins: same as start_extend_bins, how many bins to shift at the end of motif. Shifting left is negative and
                        right is positive. Default to be 0.
    
    '''
    bird_folder_data, bird_folder_save, repos_folder, results_folder = locate_folders(bird_id, model_name)
    save_name = 'mel_%02d_%s.p' %(num_lookbacks, labels)
    
    fft_size = bin_size*16 # window size for the FFT
    step_size = bin_size # distance to slide along the window (in time)
    spec_thresh = 5 # threshold for spectrograms (lower filters out more noise)
    lowcut = 500 # Hz # Low cut for our butter bandpass filter
    highcut = 15000 # Hz # High cut for our butter bandpass filter
    # For mels
    # number of mel frequency channels
    shorten_factor = 1 # how much should we compress the x-axis (time)
     # Hz # What frequency to start sampling our melS from 
     # Hz # What frequency to stop sampling our melS from 
    
    with h5py.File(neural_file_name,'r') as f:
        time_samples = np.array(f['/channel_groups/0/spikes/time_samples'].value, dtype='int64')
        cluster_num = np.array(f['/channel_groups/0/spikes/clusters/main'].value,dtype='int64')
        recording_num = np.array(f['/channel_groups/0/spikes/recording'].value, dtype='int64')

        cont_test = np.array(recording_num[:-1], dtype='int64')-np.array(recording_num[1:], dtype='int64')
        errors = [i for i, j in enumerate(cont_test) if j>0]
        if errors:
            for error_i in errors:
                if time_samples[error_i]<time_samples[error_i+1]:
                    recording_num[error_i+1] = recording_num[error_i]
                else:
                    recording_num[error_i+1] = recording_num[error_i+2]
            print('***Erroneous indices fixed.')
        '''
        if max(recording_num)>20:
            error_indices = [i for i,v in enumerate(list(recording_num)) if v > 20]
            for index in error_indices:
                recording_num[index] = recording_num[index-1]
            print('***Erroneous indices fixed.')
        '''

                #print(type(time_samples))
        rec_counter = collections.Counter(recording_num)
        rec_list = rec_counter.keys()
        spike_counts = rec_counter.values()
        time_samples_each_rec = list()
        cluster_each_rec = list()

                #separate time samples based on recording
        for rec in rec_list:
                    #start_samples.append(f['/recordings/'][str(rec)].attrs['start_sample'])
            prev_num_samples = sum(spike_counts[:rec])
            #print(prev_num_samples)
            time_samples_each_rec.append(time_samples[prev_num_samples:prev_num_samples+spike_counts[rec]])
                    #time_samples[prev_num_samples:prev_num_samples+spike_counts[rec]] = time_samples[
                    #    prev_num_samples:prev_num_samples+spike_counts[rec]]+f['/recordings/'][str(rec)].attrs['start_sample']
            cluster_each_rec.append(cluster_num[prev_num_samples:prev_num_samples+spike_counts[rec]])
    
    mel_filter, mel_inversion_filter = create_mel_filter(fft_size = fft_size,
                                                        n_freq_components = n_mel_freq_components,
                                                        start_freq = start_freq,
                                                        end_freq = end_freq)
    
    pickle_data = pd.read_pickle(bird_song_file)
    indeces = [i for i, j in enumerate(pickle_data['recon_folder']) if specify_dir in j]
    indeces = [i for i, j in zip(indeces, pickle_data['syllable_labels'][indeces]) if j in labels]
    
    mel_neural_comb = list()
    
    recording_folder_save = os.path.join(bird_folder_save, specify_dir)
    if not os.path.exists(recording_folder_save):
        os.makedirs(recording_folder_save)
    
    total_bins = 0
    for ind in indeces:
        wav_name = pickle_data['recon_folder'][ind]
        rec = int(wav_name[wav_name.index('rec_')+4:wav_name.index('rec_')+7])
        ori_song_length = int(pickle_data['recon_length'][ind]*30000)
        song_length = int(np.ceil(pickle_data['recon_length'][ind]*30000/fft_size)*fft_size)
        num_bins = int(np.ceil(ori_song_length/bin_size-2))
        
        total_bins+=num_bins
        
        #song_length = int(num_bins*bin_size)
        #print(song_length)
        neural_song_length = song_length+(num_lookbacks-2)*bin_size
        
        song_ideal_start = int(pickle_data['recon_t_rel_wav'][ind]*30000)
        song_ideal_end = song_ideal_start+song_length
        
        neural_ideal_start = song_ideal_start+bin_size*(1-num_lookbacks)
        neural_ideal_end = neural_ideal_start+neural_song_length
        
        wav_rate, wav_data = wavfile.read(wav_name)
        wav_data = wav_data[song_ideal_start:song_ideal_end]
        wav_data = butter_bandpass_filter(wav_data, lowcut, highcut, wav_rate, order=1)
        
        fig = plt.figure()
        plt.plot(wav_data)
        plt.title(str(rec)+': '+str(song_ideal_start))
        plt.savefig(os.path.join(recording_folder_save, 'motif '+str(rec)+' '+str(song_ideal_start)+'.png'))
        plt.close(fig)
        
        wav_spectrogram = pretty_spectrogram(wav_data.astype('float64'), fft_size = fft_size, 
                                   step_size = step_size, log = True, thresh = spec_thresh)
        mel_list = list(make_mel(wav_spectrogram, mel_filter, shorten_factor = shorten_factor).transpose())
                
        current_t_list = time_samples_each_rec[rec] #extract neural activities for current motif
        current_cluster_list = cluster_each_rec[rec]  #extract cluster info for current motif

        neural_start_index = bisect.bisect_left(current_t_list, neural_ideal_start) #locate the index of starting point of the current motif
        neural_start_val = current_t_list[neural_start_index] #the actual starting point of the current motif
                #start_empty_bins = (neural_start_val-neural_ideal_start)//bin_size
                
        neural_end_index = bisect.bisect_left(current_t_list, neural_ideal_end)-1 #the end index of the current motif wrt indexing within the entire recording
                #end_empty_bins = (neural_ideal_end-current_t_list[neural_end_index])//bin_size

        t_during_song = current_t_list[neural_start_index:neural_end_index+1]-neural_ideal_start#normalize time sequence
        cluster_during_song = current_cluster_list[neural_start_index:neural_end_index+1] #extract cluster sequence

        counts_list = list() #contains all counts data for this motif
        for i in range(num_bins+num_lookbacks):
            bin_start_index = bisect.bisect_left(t_during_song, i*bin_size)
            bin_end_index = bisect.bisect_left(t_during_song,(i+1)*bin_size)
            counts, bins = np.histogram(cluster_during_song[bin_start_index:bin_end_index], bins=np.arange(0,num_clusters+1))
            counts_list.append(counts)
            
        fig = plt.figure()
        plt.plot(np.sum(np.array(counts_list), axis=1))
        plt.title(str(rec)+': '+str(song_ideal_start))
        plt.savefig(os.path.join(recording_folder_save, 'motif '+str(rec)+' '+str(song_ideal_start)+'_neural.png'))
        plt.close(fig)
        
        for i in range(num_bins):
            mel_neural_tuple = (counts_list[i:i+num_lookbacks], mel_list[i])
            mel_neural_comb.append(mel_neural_tuple)       
   
    if total_bins != len(mel_neural_comb):
        print('Total number of bins is not correct.')
    print('Finished.')               
    pickle.dump(mel_neural_comb, open(os.path.join(bird_folder_save, save_name),'wb'))
    return mel_neural_comb
        
    


# ### Starling Spec Prediction with Stateful LSTM

# In[ ]:


def make_datasets_starling_stateful(neural_file_name, song_file_name, specify_dir, bin_size, num_clusters, num_lookbacks,
                           bird_id, model_name, labels,
                            n_mel_freq_components = 64, start_freq = 200, end_freq = 15000):
    '''
    This functionw works on starling data where song info is stored in pickle file.
    neural_file_name: kwik file that contains neural data
    song_file_name: kwik file that contains motif starting time and recordings
    song_length: length of motif
    bin_size: how many datapoints to combine into a bin
    num_clusters: how many clusters of neurons are recorded
    start_extend_bins: at the beginning of the motif, how many bins do you want to shift. Shifting the start to the left
                        is negative, and shifting to the right is positive. Default to be 0
    end_extend_bins: same as start_extend_bins, how many bins to shift at the end of motif. Shifting left is negative and
                        right is positive. Default to be 0.
    
    '''
    bird_folder_data, bird_folder_save, repos_folder, results_folder = locate_folders(bird_id, model_name)
    save_name = 'mel_%02d_%s_stateful.p' %(num_lookbacks, labels)
    
    fft_size = bin_size*16 # window size for the FFT
    step_size = bin_size # distance to slide along the window (in time)
    spec_thresh = 5 # threshold for spectrograms (lower filters out more noise)
    lowcut = 500 # Hz # Low cut for our butter bandpass filter
    highcut = 15000 # Hz # High cut for our butter bandpass filter
    # For mels
    # number of mel frequency channels
    shorten_factor = 1 # how much should we compress the x-axis (time)
    num_bins = 0
     # Hz # What frequency to start sampling our melS from 
     # Hz # What frequency to stop sampling our melS from 
    
    directory = os.path.join(bird_folder_data, specify_dir)
    time_samples_each_rec, cluster_each_rec = separate_recs(directory, neural_file_name, bird_id, model_name)
    
    recording_folder_save = os.path.join(bird_folder_save, specify_dir)
    if not os.path.exists(recording_folder_save):
        os.makedirs(recording_folder_save)
    
    mel_neural_comb = calculate_neural_mel(directory, time_samples_each_rec, cluster_each_rec, 0, 0, 
                                                recording_folder_save, 
                                                neural_file_name, song_file_name, bird_id, model_name, num_clusters, bin_size, num_bins, 
                                                num_lookbacks, n_mel_freq_components, start_freq, end_freq, start_extend_bins=0, 
                                                end_extend_bins=0, labels = labels)
             
    pickle.dump(mel_neural_comb, open(os.path.join(bird_folder_save, save_name),'wb'))
    return mel_neural_comb
        
    


# ### ZF Spec Prediction with Stateful LSTM

# In[ ]:


def make_datasets_finch_stateful(neural_file_name, song_file_name, song_length, bin_size, num_lookbacks, num_clusters, 
                             bird_id, model_name, start_extend_bins=0, 
                               end_extend_bins=0, specify_subdir=None,
                            n_mel_freq_components = 64, start_freq = 200, end_freq = 15000):
    '''
    neural_file_name: kwik file that contains neural data
    song_file_name: kwik file that contains motif starting time and recordings
    song_length: length of motif
    bin_size: how many datapoints to combine into a bin
    num_clusters: how many clusters of neurons are recorded
    start_extend_bins: at the beginning of the motif, how many bins do you want to shift. Shifting the start to the left
                        is negative, and shifting to the right is positive. Default to be 0
    end_extend_bins: same as start_extend_bins, how many bins to shift at the end of motif. Shifting left is negative and
                        right is positive. Default to be 0.
    
    '''
    #num_lookbacks = 1
    bird_folder_data, bird_folder_save, repos_folder, results_folder = locate_folders(bird_id, model_name)
    save_name = 'mel_%02d_%d_%d_stateful.p' %(num_lookbacks, start_extend_bins, end_extend_bins)
    
    start_extend_bins, end_extend_bins = cal_extend_size(bin_size, start_extend_bins, end_extend_bins)
    num_bins = cal_num_bins(song_length, bin_size, start_extend_bins, end_extend_bins)
    subdirs = which_dir(bird_folder_data, specify_subdir)
    extended_song_length, neural_song_length = cal_modified_length(song_length, bin_size, start_extend_bins, end_extend_bins, num_lookbacks) 
    
    '''
    all_batches = list()
    for _ in range(num_bins):
        all_batches.append(list())
    '''
    mel_neural_comb=list()
    
    for directory in subdirs:
        kwik_file = os.path.join(directory, neural_file_name)
        song_file = os.path.join(directory, song_file_name)
        
        try:
            trial1 = h5py.File(kwik_file,'r')['/channel_groups/0/spikes/time_samples'].value
        except IOError:
            print('--->Skipping '+directory+' because it doesnt contain spiking kwik files.')
            continue
            
        if "sleep" in directory:
            print('--->Skipping '+directory+' because it is a sleep recording.')
            continue
        
        recording = directory[directory.index('day-'):]
        recording_folder_save = os.path.join(bird_folder_save, recording)
        if not os.path.exists(recording_folder_save):
            os.makedirs(recording_folder_save)
        
        time_samples_each_rec, cluster_each_rec = separate_recs(directory, neural_file_name, bird_id, model_name)
            
        mel_neural_comb_dir = calculate_neural_mel(directory, time_samples_each_rec, cluster_each_rec, extended_song_length, neural_song_length, 
                                                     recording_folder_save, 
                                                     neural_file_name, song_file_name, bird_id, model_name, num_clusters, bin_size, num_bins, 
                                                     num_lookbacks, n_mel_freq_components, start_freq, end_freq, start_extend_bins, 
                                                     end_extend_bins)
        mel_neural_comb += mel_neural_comb_dir
        #mel_neural_comb is a collection of lists, each inidividual list is a song, contains a colletion of neural-mel 
        #tuples within a song
        
        print('Finished.')
        
    #regoup into batch based   
    pickle.dump(mel_neural_comb, open(os.path.join(bird_folder_save, save_name),'wb'))
        
    return mel_neural_comb


# ### ZF Spec Prediction with Feed Forward NN

# In[ ]:


def make_datasets_finch_ffnn(neural_file_name, song_file_name, song_length, bin_size, num_lookbacks, num_clusters, 
                             bird_id, model_name, start_extend_bins=0, 
                               end_extend_bins=0, specify_subdir=None,
                            n_mel_freq_components = 64, start_freq = 200, end_freq = 15000, slice_shuffle_pattern=None):
    '''
    neural_file_name: kwik file that contains neural data
    song_file_name: kwik file that contains motif starting time and recordings
    song_length: length of motif
    bin_size: how many datapoints to combine into a bin
    num_clusters: how many clusters of neurons are recorded
    start_extend_bins: at the beginning of the motif, how many bins do you want to shift. Shifting the start to the left
                        is negative, and shifting to the right is positive. Default to be 0
    end_extend_bins: same as start_extend_bins, how many bins to shift at the end of motif. Shifting left is negative and
                        right is positive. Default to be 0.
    
    '''
    #num_lookbacks = 1
    bird_folder_data, bird_folder_save, repos_folder, results_folder = locate_folders(bird_id, model_name)
    save_name = 'mel_%02d_%d_%d_ffnn.p' %(num_lookbacks, start_extend_bins, end_extend_bins)
    
    start_extend_bins, end_extend_bins = cal_extend_size(bin_size, start_extend_bins, end_extend_bins)
    num_bins = cal_num_bins(song_length, bin_size, start_extend_bins, end_extend_bins)
    subdirs = which_dir(bird_folder_data, specify_subdir)
    extended_song_length, neural_song_length = cal_modified_length(song_length, bin_size, start_extend_bins, end_extend_bins, num_lookbacks) 
    
    '''
    all_batches = list()
    for _ in range(num_bins):
        all_batches.append(list())
    '''
    mel_neural_comb=list()
    
    for directory in subdirs:
        kwik_file = os.path.join(directory, neural_file_name)
        song_file = os.path.join(directory, song_file_name)
        
        try:
            trial1 = h5py.File(kwik_file,'r')['/channel_groups/0/spikes/time_samples'].value
        except IOError:
            print('--->Skipping '+directory+' because it doesnt contain spiking kwik files.')
            continue
            
        if "sleep" in directory:
            print('--->Skipping '+directory+' because it is a sleep recording.')
            continue
        
        recording = directory[directory.index('day-'):]
        recording_folder_save = os.path.join(bird_folder_save, recording)
        if not os.path.exists(recording_folder_save):
            os.makedirs(recording_folder_save)
        
        time_samples_each_rec, cluster_each_rec = separate_recs(directory, neural_file_name, bird_id, model_name)
            
        mel_neural_comb_dir = calculate_neural_mel(directory, time_samples_each_rec, cluster_each_rec, extended_song_length, neural_song_length, 
                                                     recording_folder_save, slice_shuffle_pattern,
                                                     neural_file_name, song_file_name, bird_id, model_name, num_clusters, bin_size, num_bins, 
                                                     num_lookbacks, n_mel_freq_components, start_freq, end_freq, start_extend_bins, 
                                                     end_extend_bins, flatten_inputs = True)

        mel_neural_comb += mel_neural_comb_dir
        #mel_neural_comb is a collection of lists, each inidividual list is a song, contains a colletion of neural-mel 
        #tuples within a song
        
        print('Finished.')
        
    mel_neural_comb = song_based_to_all(mel_neural_comb)
        
    #regoup into batch based   
    pickle.dump(mel_neural_comb, open(os.path.join(bird_folder_save, save_name),'wb'))
        
    return mel_neural_comb


# ### Spike Counts Prediction with Spectrogram

# In[ ]:


def make_datasets_spec2neuro(neural_file_name, song_file_name, song_length, bin_size, num_lookbacks, num_clusters, 
                             bird_id, model_name, start_extend_bins=0, 
                               end_extend_bins=0, specify_subdir=None,
                            n_mel_freq_components = 64, start_freq = 200, end_freq = 15000):
    '''
    neural_file_name: kwik file that contains neural data
    song_file_name: kwik file that contains motif starting time and recordings
    song_length: length of motif
    bin_size: how many datapoints to combine into a bin
    num_clusters: how many clusters of neurons are recorded
    start_extend_bins: at the beginning of the motif, how many bins do you want to shift. Shifting the start to the left
                        is negative, and shifting to the right is positive. Default to be 0
    end_extend_bins: same as start_extend_bins, how many bins to shift at the end of motif. Shifting left is negative and
                        right is positive. Default to be 0.
    
    '''
    #num_lookbacks = 1
    bird_folder_data, bird_folder_save, repos_folder, results_folder = locate_folders(bird_id, model_name)
    save_name = 'mel_%02d_%d_%d_spec2neuro.p' %(num_lookbacks, start_extend_bins, end_extend_bins)
    
    start_extend_bins, end_extend_bins = cal_extend_size(bin_size, start_extend_bins, end_extend_bins)
    num_bins = cal_num_bins(song_length, bin_size, start_extend_bins, end_extend_bins)
    subdirs = which_dir(bird_folder_data, specify_subdir)
    extended_song_length, neural_song_length = cal_modified_length(song_length, bin_size, start_extend_bins, 
                                                                   end_extend_bins, num_lookbacks, mode='spec2neuro') 
    
    '''
    all_batches = list()
    for _ in range(num_bins):
        all_batches.append(list())
    '''
    mel_neural_comb=list()
    
    for directory in subdirs:
        kwik_file = os.path.join(directory, neural_file_name)
        song_file = os.path.join(directory, song_file_name)
        
        try:
            trial1 = h5py.File(kwik_file,'r')['/channel_groups/0/spikes/time_samples'].value
        except IOError:
            print('--->Skipping '+directory+' because it doesnt contain spiking kwik files.')
            continue
            
        if "sleep" in directory:
            print('--->Skipping '+directory+' because it is a sleep recording.')
            continue
        
        recording = directory[directory.index('day-'):]
        recording_folder_save = os.path.join(bird_folder_save, recording)
        if not os.path.exists(recording_folder_save):
            os.makedirs(recording_folder_save)
        
        time_samples_each_rec, cluster_each_rec = separate_recs(directory, neural_file_name, bird_id, model_name)
            
        mel_neural_comb_dir = calculate_neural_mel(directory, time_samples_each_rec, cluster_each_rec, extended_song_length, neural_song_length, 
                                                     recording_folder_save, 
                                                     neural_file_name, song_file_name, bird_id, model_name, num_clusters, bin_size, num_bins, 
                                                     num_lookbacks, n_mel_freq_components, start_freq, end_freq, start_extend_bins, 
                                                     end_extend_bins, mode='spec2neuro')
        mel_neural_comb += mel_neural_comb_dir
        #mel_neural_comb is a collection of lists, each inidividual list is a song, contains a colletion of neural-mel 
        #tuples within a song
        
        print('Finished.')
        
    #regoup into batch based   
    pickle.dump(mel_neural_comb, open(os.path.join(bird_folder_save, save_name),'wb'))
        
    return mel_neural_comb


# ### ZF Spec Prediction with Latency

# In[ ]:


def make_datasets_finch_latency(neural_file_name, song_file_name, song_length, bin_size, num_clusters, num_lookbacks, 
                                latency, bird_id, model_name, 
                                start_extend_bins=0, end_extend_bins=0, specify_subdir=None, n_mel_freq_components = 64, 
                                start_freq = 200, end_freq = 15000):
    '''
    neural_file_name: kwik file that contains neural data
    song_file_name: kwik file that contains motif starting time and recordings
    song_length: length of motif
    bin_size: how many datapoints to combine into a bin
    num_clusters: how many clusters of neurons are recorded
    start_extend_bins: at the beginning of the motif, how many bins do you want to shift. Shifting the start to the left
                        is negative, and shifting to the right is positive. Default to be 0
    end_extend_bins: same as start_extend_bins, how many bins to shift at the end of motif. Shifting left is negative and
                        right is positive. Default to be 0.
    
    '''
    bird_folder_data, bird_folder_save, repos_folder, results_folder = locate_folders(bird_id, model_name)
    save_name = 'mel_%02d_%d_%d.p' %(num_lookbacks, start_extend_bins, end_extend_bins)
    
    fft_size = bin_size*16 # window size for the FFT
    step_size = bin_size # distance to slide along the window (in time)
    start_extend_bins *= int(fft_size/bin_size)
    end_extend_bins *= int(fft_size/bin_size)
    spec_thresh = 5 # threshold for spectrograms (lower filters out more noise)
    lowcut = 500 # Hz # Low cut for our butter bandpass filter
    highcut = 15000 # Hz # High cut for our butter bandpass filter
    # For mels
    # number of mel frequency channels
    shorten_factor = 1 # how much should we compress the x-axis (time)
     # Hz # What frequency to start sampling our melS from 
     # Hz # What frequency to stop sampling our melS from 
    
    mel_filter, mel_inversion_filter = create_mel_filter(fft_size = fft_size,
                                                        n_freq_components = n_mel_freq_components,
                                                        start_freq = start_freq,
                                                        end_freq = end_freq)
    
    num_bins = int(song_length/bin_size-2-start_extend_bins+end_extend_bins)#number of bins in each song
    
    if specify_subdir:
        subdirs = [os.path.join(bird_folder_data, specify_subdir)]
    else:
        subdirs = [combined[0] for combined in os.walk(bird_folder_data)][1:]
        
    extended_song_length = song_length+bin_size*(end_extend_bins-start_extend_bins)+fft_size
    neural_song_length = extended_song_length+(num_lookbacks-2)*bin_size 
    #take out two bins due to mel processing algorithm
    
    mel_neural_comb = list()
    tot_motifs = 0
    
    for directory in subdirs:
        kwik_file = os.path.join(directory, neural_file_name)
        song_file = os.path.join(directory, song_file_name)
        
        try:
            trial1 = h5py.File(kwik_file,'r')['/channel_groups/0/spikes/time_samples'].value
        except IOError:
            print('--->Skipping '+directory+' because it doesnt contain spiking kwik files.')
            continue
            
        if "sleep" in directory:
            print('--->Skipping '+directory+' because it is a sleep recording.')
            continue
            
        recording = directory[directory.index('day-'):]
        recording_folder_save = os.path.join(bird_folder_save, recording)
        if not os.path.exists(recording_folder_save):
            os.makedirs(recording_folder_save)
            
        print('Start working on '+directory)
        
        with h5py.File(kwik_file,'r') as f:
            time_samples = f['/channel_groups/0/spikes/time_samples'].value
            cluster_num = f['/channel_groups/0/spikes/clusters/main'].value
            recording_num = f['/channel_groups/0/spikes/recording'].value
            if max(recording_num)>20:
                error_indices = [i for i,v in enumerate(list(recording_num)) if v > 20]
                for index in error_indices:
                    recording_num[index] = recording_num[index-1]
                print('***Erroneous indices fixed.')
                
            #print(type(time_samples))
            rec_counter = collections.Counter(recording_num)
            rec_list = rec_counter.keys()
            spike_counts = rec_counter.values()
            time_samples_each_rec = list()
            cluster_each_rec = list()

            #separate time samples based on recording
            for rec in rec_list:
                #start_samples.append(f['/recordings/'][str(rec)].attrs['start_sample'])
                prev_num_samples = sum(spike_counts[:rec])
                time_samples_each_rec.append(time_samples[prev_num_samples:prev_num_samples+spike_counts[rec]])
                #time_samples[prev_num_samples:prev_num_samples+spike_counts[rec]] = time_samples[
                #    prev_num_samples:prev_num_samples+spike_counts[rec]]+f['/recordings/'][str(rec)].attrs['start_sample']
                cluster_each_rec.append(cluster_num[prev_num_samples:prev_num_samples+spike_counts[rec]])

            #print(np.shape(time_samples_each_rec))
            #print(time_samples_each_rec)
            
        with h5py.File(song_file, 'r') as f:
            motif_starts = f['event_types/singing/motiff_1/time_samples'].value  #start time of each motif
            motif_recs = f['event_types/singing/motiff_1/recording'].value  #recording number of each motif
            songtuples = zip(motif_recs, motif_starts)
            
            tot_motifs += len(songtuples)

            for rec, ideal_start in songtuples:
                
                neural_ideal_start = ideal_start+bin_size*(start_extend_bins+1-num_lookbacks-latency)
                neural_ideal_end = neural_ideal_start+neural_song_length
                
                song_ideal_start = ideal_start+bin_size*start_extend_bins
                song_ideal_end = song_ideal_start+extended_song_length
                
                #process wav file into mel spec
                wav_name = 'experiment-rec_0%02d.mic.wav' % (rec)
                wav_file = os.path.join(directory, wav_name)
                
                wav_rate, wav_data = wavfile.read(wav_file)
                wav_data = wav_data[song_ideal_start:song_ideal_end]
                wav_data = butter_bandpass_filter(wav_data, lowcut, highcut, wav_rate, order=1)
                
                if len(wav_data)!=extended_song_length:
                    raise ValueError('less wav datapointes loaded than expected')
                
                #plot each filtered unmel-edwav
                fig = plt.figure()
                plt.plot(wav_data)
                plt.title(str(rec)+': '+str(song_ideal_start))
                plt.savefig(os.path.join(recording_folder_save, 'motif '+str(rec)+' '+str(ideal_start)+'.png'))
                plt.close(fig)
                
                #mel and compile into a list based on time bins
                _, _, wav_spectrogram = pretty_spectrogram_zeke(wav_data.astype('float64'), wav_rate, thresh = spec_thresh, 
                                                            fft_size=fft_size, log=True, step_size=step_size, 
                                                            window=('gaussian', 80), f_min=0, f_max=15000)
                #print(wav_spectrogram)
                mel_array = make_mel(wav_spectrogram.transpose(), mel_filter, shorten_factor = shorten_factor).transpose()
                #print(mel_array.shape)
                mel_list = list(mel_array)
                #print(len(mel_list))
                
                fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(8,4))
                cax = ax.matshow(np.array(mel_list, dtype = 'float64').transpose(), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
                fig.colorbar(cax)
                plt.savefig(os.path.join(recording_folder_save, 'motif '+str(rec)+' '+str(ideal_start)+'_mel.png'))
                plt.close(fig)
                
                current_t_list = time_samples_each_rec[rec] #extract neural activities for current motif
                current_cluster_list = cluster_each_rec[rec]  #extract cluster info for current motif

                neural_start_index = bisect.bisect_left(current_t_list, neural_ideal_start) #locate the index of starting point of the current motif
                neural_start_val = current_t_list[neural_start_index] #the actual starting point of the current motif
                #start_empty_bins = (neural_start_val-neural_ideal_start)//bin_size
                
                neural_end_index = bisect.bisect_left(current_t_list, neural_ideal_end)-1 #the end index of the current motif wrt indexing within the entire recording
                #end_empty_bins = (neural_ideal_end-current_t_list[neural_end_index])//bin_size

                t_during_song = current_t_list[neural_start_index:neural_end_index+1]-neural_ideal_start#normalize time sequence
                cluster_during_song = current_cluster_list[neural_start_index:neural_end_index+1] #extract cluster sequence

                counts_list = list() #contains all counts data for this motif
                for i in range(num_bins+num_lookbacks):
                    bin_start_index = bisect.bisect_left(t_during_song, i*bin_size)
                    bin_end_index = bisect.bisect_left(t_during_song,(i+1)*bin_size)
                    counts, bins = np.histogram(cluster_during_song[bin_start_index:bin_end_index], bins=np.arange(0,num_clusters+1))
                    counts_list.append(counts)
                    
                fig = plt.figure()
                plt.plot(np.sum(np.array(counts_list), axis=1))
                plt.title(str(rec)+': '+str(song_ideal_start))
                plt.savefig(os.path.join(recording_folder_save, 'motif '+str(rec)+' '+str(ideal_start)+'_neural.png'))
                plt.close(fig)    
                
                for i in range(num_bins):
                    mel_neural_tuple = (counts_list[i:i+num_lookbacks], mel_list[i])
                    mel_neural_comb.append(mel_neural_tuple)

                """for cluster in range(num_clusters):
                    index_this_cluster = [i for i, j in enumerate(cluster_during_song) if j == cluster]
                    t_this_cluster = t_during_song[index_this_cluster]
                    counts,bins = np.histogram(t_this_cluster, bins=[start_val:bin_size:end_val, sys.maxint])
                    """

                if counts_list[0].shape[0] != num_clusters:
                    raise ValueError('check cluster histogram')
                    
                #if len(counts_list)-num_lookbacks != len(mel_list):
                #    raise ValueError('Neural spiking bins do not correspond to spectrogram bins.')

                    #check for counting errors
        print('Finished.')
        
    if len(mel_neural_comb)!= num_bins*tot_motifs:
        raise ValueError('Might be missing some motifs or bins')
        
    pickle.dump(mel_neural_comb, open(os.path.join(bird_folder_save, save_name),'wb'))
        
    return mel_neural_comb


# ### Spec Prediction with Warping

# In[5]:


def make_datasets_warp(neural_file_name, song_file_name, song_length, bin_size, num_clusters, num_lookbacks, bird_id, model_name,
                        start_extend_bins=0, end_extend_bins=0, specify_subdir=None,
                        n_mel_freq_components = 64, start_freq = 200, end_freq = 15000, slice_shuffle_pattern=None, plot_raster=True):
    '''
    neural_file_name: kwik file that contains neural data
    song_file_name: kwik file that contains motif starting time and recordings
    song_length: length of motif
    bin_size: how many datapoints to combine into a bin
    num_clusters: how many clusters of neurons are recorded
    num_lookbacks: how many lookback bins the model will use as input features
    start_extend_bins: at the beginning of the motif, how many bins do you want to shift. Shifting the start to the left
                        is negative, and shifting to the right is positive. Default to be 0
    end_extend_bins: same as start_extend_bins, how many bins to shift at the end of motif. Shifting left is negative and
                        right is positive. Default to be 0.
    start_freq, end_freq: mel scale min and max frequencies.
    slice_shuffle_pattern: input the specific shuffle pattern as list, or "True" if no specific pattern and the model will generate
                        a random pattern.
    
    '''
    if slice_shuffle_pattern and (start_extend_bins or end_extend_bins):
        raise ValueError('Slice shuffling algorithm does not support extend bins now.')
    
    bird_folder_data, bird_folder_save, repos_folder, results_folder = locate_folders(bird_id, model_name)
    save_name = 'mel_%02d_%d_%d_warp.p' %(num_lookbacks, start_extend_bins, end_extend_bins)
    
    start_extend_bins, end_extend_bins = cal_extend_size(bin_size, start_extend_bins, end_extend_bins)
    num_bins = cal_num_bins(song_length, bin_size, start_extend_bins, end_extend_bins)
    subdirs = which_dir(bird_folder_data, specify_subdir)
    extended_song_length, neural_song_length = cal_modified_length(song_length, bin_size, start_extend_bins, end_extend_bins, num_lookbacks) 
    
    mel_neural_comb=list()
    
    slice_shuffle_pattern, automatic_shuffle = make_shuffle_pattern(slice_shuffle_pattern, num_bins, n_mel_freq_components)
    
    for directory in subdirs:
        kwik_file = os.path.join(directory, neural_file_name)
        song_file = os.path.join(directory, song_file_name)
        
        try:
            trial1 = h5py.File(kwik_file,'r')['/channel_groups/0/spikes/time_samples'].value
        except IOError:
            print('--->Skipping '+directory+' because it doesnt contain spiking kwik files.')
            continue
            
        if "sleep" in directory:
            print('--->Skipping '+directory+' because it is a sleep recording.')
            continue
        
        recording = directory[directory.index('day-'):]
        recording_folder_save = os.path.join(bird_folder_save, recording)
        if not os.path.exists(recording_folder_save):
            os.makedirs(recording_folder_save)
        
        time_samples_each_rec, cluster_each_rec = separate_recs(directory, neural_file_name, bird_id, model_name)
            
        mel_neural_comb_dir = calculate_neural_mel_warp(directory, time_samples_each_rec, cluster_each_rec,
                                                    recording_folder_save, slice_shuffle_pattern,
                                                    neural_file_name, song_file_name, bird_id, model_name, num_clusters, bin_size, num_bins, 
                                                    extended_song_length, num_lookbacks, n_mel_freq_components, start_freq, end_freq, start_extend_bins, 
                                                    end_extend_bins, fft_size = 256, mode = 'wav', plot_raster=plot_raster)
        
        mel_neural_comb += mel_neural_comb_dir
        #mel_neural_comb is a collection of lists, each inidividual list is a song, contains a colletion of neural-mel 
        #tuples within a song
        
        print('Finished.')
        
    mel_neural_comb = song_based_to_all(mel_neural_comb)
        
    #regoup into batch based   
    pickle.dump(mel_neural_comb, open(os.path.join(bird_folder_save, save_name),'wb'))
    
    if automatic_shuffle:
        return mel_neural_comb, slice_shuffle_pattern
    else:
        return mel_neural_comb


# In[ ]:


def warp(heads, tails, extended_song_length, bin_size, num_lookbacks, song_start, wav_data_rec, 
        temp_turns, temp_turns_re):
    turns = [0]+sorted(heads+tails)+[extended_song_length]
    song_end = song_start+extended_song_length
    segments = turns2segs(turns)
    adj_len = int((16-num_lookbacks)%16)*bin_size
    
    if not temp_turns:
        temp_heads = heads
        temp_tails = tails
        temp_turns = turns
        temp_segments = segments

        temp_offset = temp_segments[0]-(temp_segments[0]/bin_size*bin_size)
        temp_segments_ad = temp_segments[:]
        temp_segments_ad[0] -= temp_offset
        temp_segments_ad[-1] += temp_offset
        temp_segments = temp_segments_ad[:]
        temp_turns_ad = [sum(temp_segments[:i]) for i in range(len(temp_segments)+1)]
        temp_turns = temp_turns_ad[:]

        #rectify to fit into bins
        temp_turns_re = [round(float(turn)/bin_size)*bin_size for turn in temp_turns]
        temp_segments_re = turns2segs(temp_turns_re)

        aligned_start = song_start+temp_offset
        aligned_end = song_end+temp_offset

        w_wav = wav_data_rec[aligned_start:aligned_end+adj_len]
        if len(w_wav)!=extended_song_length+adj_len:
            print('template wav has a different size')
        w_binturns = np.arange(temp_turns_re[0]+aligned_start-num_lookbacks*bin_size+bin_size, 
                               temp_turns_re[-1]+aligned_start, bin_size)
        
        return([temp_turns, temp_turns_re, aligned_start, aligned_end, w_wav, w_binturns])

    else:
        temp_segments = turns2segs(temp_turns)
        temp_segments_re = turns2segs(temp_turns_re)
        head_offset = segments[0]-temp_segments[0]
        tail_offset = segments[-1]-temp_segments[-1]
        aligned_start = song_start+head_offset
        aligned_end = song_end-tail_offset
        aligned_wav = wav_data_rec[aligned_start:aligned_end+adj_len]
        new_turns = [turns[0]]+[turn-head_offset for turn in turns[1:]]
        w_wav = list()
        w_binturns = list()
        new_turns[-1] = new_turns[-1]-tail_offset
        new_segments = turns2segs(new_turns)

        if aligned_end-aligned_start!=new_turns[-1]-new_turns[0]:
            raise ValueError('check turn alignment')

        new_turns_re = [0]
        #resample segments of wav file to get the new wav
        for i in range(len(new_segments)):
            w_wav+=list(sp.signal.resample(aligned_wav[new_turns[i]:new_turns[i+1]], temp_segments[i]))
            new_turns_re.append((temp_turns_re[i+1]-temp_turns[i+1])/temp_segments[i]*new_segments[i]+new_turns[i+1])
        w_wav+=list(wav_data_rec[aligned_end:aligned_end+adj_len])
        if len(w_wav)!=extended_song_length+adj_len:
            raise ValueError('resampled wav has a different size')
        new_turns_re[-2]=new_turns_re[-1]-temp_segments_re[-1]
        new_segments_re = turns2segs(new_turns_re)
        
        #w_binturns are absolute time
        w_binturns+=list(np.arange(aligned_start-num_lookbacks*bin_size+bin_size, aligned_start, bin_size))
        for i in range(len(new_segments_re)):
            new_bin_size = bin_size*new_segments_re[i]/temp_segments_re[i]
            if i == len(new_segments_re)-1:
                if new_bin_size!=bin_size:
                    raise ValueError('check bin_size calculation')
                w_binturns+=list(np.arange(new_turns_re[i]+aligned_start, new_turns_re[i+1]+aligned_start+new_bin_size, new_bin_size))
            else:
                w_binturns+=list(np.arange(new_turns_re[i]+aligned_start, new_turns_re[i+1]+aligned_start, new_bin_size))
        #take out potential duplicates
        no_duplicate = [w_binturns[i] for i in range(len(w_binturns)-1) if abs(w_binturns[i+1]-w_binturns[i])>10]
        w_binturns = no_duplicate[:]
        
        print(len(w_binturns))
        
        return([new_turns, new_turns_re, aligned_start, aligned_end, w_wav, w_binturns])


# In[ ]:


def calculate_neural_mel_warp(directory, time_samples_each_rec, cluster_each_rec,
                        recording_folder_save, slice_shuffle_pattern,
                        neural_file_name, song_file_name, bird_id, model_name, num_clusters, bin_size, num_bins, 
                        extended_song_length, num_lookbacks, n_mel_freq_components, start_freq, end_freq, start_extend_bins, 
                        end_extend_bins, fft_size = 256, mode = 'fft', blanks_ratio = 0, plot_raster=True):
    
    start_extend_pts = bin_size*start_extend_bins
    end_extend_pts = bin_size*end_extend_bins
    
    if song_file_name.endswith('kwik') or song_file_name.endswith('kwe'):
        song_file_type = 'kwik'
    else:
        song_file_type = 'pickle'
    song_file = os.path.join(directory, song_file_name)
    
    fig_folder_save = os.path.join(recording_folder_save, 'warped', '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now()))
    
    if not os.path.exists(fig_folder_save):
        os.makedirs(fig_folder_save)
        
    mel_neural_comb = list()
    count = 0
    
    if song_file_type=='kwik':
        motif_starts = h5py.File(song_file, 'r')['event_types/singing/motiff_1/time_samples'].value  #start time of each motif
        motif_recs = h5py.File(song_file, 'r')['event_types/singing/motiff_1/recording'].value  #recording number of each motif
        songtuples = zip(motif_recs, motif_starts)
        
        last_rec = -1
        if blanks_ratio:
            songtuples += generate_blanks(songtuples, extended_song_length, ratio = blanks_ratio)
        
        positions = [[] for i in range(num_clusters)]
        spec_list = list()
        wav_list = list()
            
        for rec, ideal_start in songtuples:
            
            mel_neural_comb_song = list()

            song_start = ideal_start-start_extend_pts
            song_end = song_start+extended_song_length
            warp_end = song_end-cal_fft_size(bin_size)
                
            #process wav file into mel spec
            
            step_size = bin_size # distance to slide along the window (in time)
            spec_thresh = 5 # threshold for spectrograms (lower filters out more noise)
            lowcut = 500 # Hz # Low cut for our butter bandpass filter
            highcut = 14999 # Hz # High cut for our butter bandpass filter
            # For mels
            shorten_factor = 1 # how much should we compress the x-axis (time)
            
            if last_rec != rec:
                wav_name = 'experiment-rec_0%02d.mic.wav' % (rec)
                wav_file = os.path.join(directory, wav_name)
                wav_rate, wav_data_rec = wavfile.read(wav_file)
                wav_data_rec = butter_bandpass_filter(wav_data_rec, lowcut, highcut, wav_rate, order=1)
                last_rec = rec
            
            fig1 = plt.figure()
            plt.plot(wav_data_rec[song_start:song_end])
            plt.title(str(rec)+': '+str(song_start))
            plt.savefig(os.path.join(fig_folder_save, '%03d_original.png' %(count)))
            plt.close(fig1)
            
            #here we find heads, tails, lengths in terms of points, aka step size is set to 1
            heads, tails, lengths = find_syllable_boundaries(wav_data_rec, song_start, song_end, wav_rate, fft_size, 1, count, 
                                                             fig_folder_save = fig_folder_save, mode=mode)
            
            if len(heads)!=5:
                continue
                
            #segment lengths, pattern should be silence, syllable, silence, syllable...
            
            if not count:
                temp_turns = None
                temp_turns_re = None
                
                temp_turns, temp_turns_re, aligned_start, aligned_end, w_wav, w_binturns = warp(heads, tails, extended_song_length, 
                                                                                                bin_size, num_lookbacks, song_start, 
                                                                                                wav_data_rec, temp_turns, temp_turns_re)
            else:
                
                new_turns, new_turns_re, aligned_start, aligned_end, w_wav, w_binturns = warp(heads, tails, extended_song_length, 
                                                                                                bin_size, num_lookbacks, song_start, 
                                                                                                wav_data_rec, temp_turns, temp_turns_re)

            #plot each filtered unmel-edwav
            fig2 = plt.figure()
            plt.plot(w_wav)
            plt.title(str(rec)+': '+str(aligned_start))
            plt.savefig(os.path.join(fig_folder_save, '%03d.png' %(count)))
            plt.close(fig2)
            
            wav_list.append(w_wav)
            
            mel_list = make_specs(w_wav, rec, aligned_start, aligned_end, fig_folder_save, bin_size, n_mel_freq_components, 
                                  start_freq, end_freq)
            
            spec_list.append(np.array(mel_list))
            
            current_t_list = time_samples_each_rec[rec] #extract neural activities for current motif
            current_cluster_list = cluster_each_rec[rec]  #extract cluster info for current motif

            counts_list = spike_counter_warp(current_t_list, current_cluster_list, w_binturns, rec,
                                              fig_folder_save, count, 
                                              num_clusters, num_bins, num_lookbacks, bin_size, plot_raster=plot_raster)
            
            if plot_raster:
                start_index = bisect.bisect_left(current_t_list, aligned_start+bin_size)
                end_index = bisect.bisect_left(current_t_list, aligned_end-bin_size)
                t_list = current_t_list[start_index:end_index]
                bin_start_index = bisect.bisect_left(w_binturns, aligned_start+bin_size)
                bin_end_index = bisect.bisect_left(w_binturns, aligned_end-bin_size)
                
                if w_binturns[bin_start_index]!=aligned_start+bin_size:
                    print('Misaligned start:'+ str(w_binturns[bin_start_index])+' '+str(aligned_start+bin_size))
                if w_binturns[bin_end_index]!=aligned_end-bin_size:
                    print('Misaligned end:'+ str(w_binturns[bin_end_index])+' '+str(aligned_end-bin_size))
                    
                binturns_during_song = w_binturns[bin_start_index:bin_end_index+1]
                new_t_list = warp_spike_train(binturns_during_song, bin_size, t_list)

                this_time = [int(t-w_binturns[bin_start_index]) for t in new_t_list]
                this_cluster = current_cluster_list[start_index:end_index]
                for c in range(num_clusters):
                    positions[c].append(np.array([this_time[i] for i, x in enumerate(this_cluster) if x == c]))
                        
            for i in range(num_bins):
                inputs = np.array(counts_list[i:i+num_lookbacks])
                outputs = np.array(mel_list[i])
                if slice_shuffle_pattern:
                    this_slice = slice_shuffle_pattern[i]
                    outputs = [outputs[i] for i in this_slice]
                mel_neural_tuple = (inputs, outputs)

                mel_neural_comb_song.append(mel_neural_tuple)
                    
            if counts_list[0].shape[0] != num_clusters:
                raise ValueError('check cluster histogram')
            mel_neural_comb.append(mel_neural_comb_song)
            count += 1
        if plot_raster:
            np.save(os.path.join(fig_folder_save, 'positions.npy'), positions)
            #raster_plotter(positions, num_clusters, fig_folder_save)
        
        spec_file = os.path.join(fig_folder_save, 'specs.npy')
        np.save(spec_file, np.array(spec_list))
        wav_file = os.path.join(fig_folder_save, 'wavs.npy')
        np.save(wav_file, np.array(wav_list))
    return mel_neural_comb


# In[ ]:


def turns2segs(turns):
    segs = [end-start for start, end in zip(turns[:-1], turns[1:])]
    return segs


# In[ ]:


def spike_counter_warp(current_t_list, current_cluster_list, w_binturns, rec,
                  recording_folder_save, count,
                  num_clusters, num_bins, num_lookbacks, bin_size, plot_raster=True):
    
    #w_binturns can be [starts, ends], or [turn[0], turn[1]...]
    if len(w_binturns) == 2:
        w_binstarts = w_binturns[0]
        w_binends = w_binturns[1]
    else:
        w_binstarts = w_binturns[:-1]
        w_binends = w_binturns[1:]
    counts_list = list() #contains all counts data for this motif
    for i in range(num_bins+num_lookbacks):
        bin_start_index = bisect.bisect_left(current_t_list, w_binstarts[i])
        bin_end_index = bisect.bisect_left(current_t_list, w_binends[i])
        counts, bins = np.histogram(current_cluster_list[bin_start_index:bin_end_index], bins=np.arange(0,num_clusters+1))
        counts_list.append(counts)
    '''
    if plot_raster:
        if plot_raster=='full':
            start_index = bisect.bisect_left(current_t_list, w_binturns[0])
            end_index = bisect.bisect_left(current_t_list, w_binturns[-1])
            t_list = current_t_list[start_index:end_index]
            new_t_list = warp_spike_train(w_binturns, bin_size, t_list)
            
            this_time = [int(t-w_binturns[0]) for t in new_t_list]
            this_cluster = current_cluster_list[start_index:end_index]
            positions = [[] for i in range(num_clusters)]
            for c in range(num_clusters):
                indices = [i for i, x in enumerate(this_cluster) if x == c]
                for
                positions.append([this_time[i] for i in indices])
           
            plt.figure(figsize=(30,20))
            plt.eventplot(np.array(positions))
            plt.title(str(rec)+': '+str(w_binturns[num_lookbacks]))
            plt.savefig(os.path.join(recording_folder_save, '%03d_raster.png' %(count)))
            plt.close('all')
            
        else:
            fig = plt.figure()
            plt.plot(np.sum(np.array(counts_list), axis=1))
            plt.title(str(rec)+': '+str(w_binturns[num_lookbacks]))
            plt.savefig(os.path.join(recording_folder_save, '%03d_neural.png' %(count)))
            plt.close(fig)

            fig = plt.imshow(np.array(counts_list).transpose())
            plt.title(str(rec)+': '+str(w_binturns[num_lookbacks]))
            plt.savefig(os.path.join(recording_folder_save, '%03d_raster.png' %(count)))
            plt.close('all')
    '''
    return counts_list


# In[ ]:


def warp_spike_train(binturns, bin_size, t_list, warp_method = 'chunks'):
    #binturns t_list are absolute time
    
    new_t_list = list()
    if warp_method == 'anderson':
        if len(binturns)!= 2:
            raise ValueError('Combine start and end when warping spike train!')
        bin_starts = binturns[0]
        bin_ends = binturns[1]
        for t in t_list:
            t_index_s = bisect.bisect_right(bin_starts, t)
            t_index_e = bisect.bisect_left(bin_ends, t)
            num_bins_before = t_index_e
            if t_index_s-t_index_e == 0:
                new_t = round(bin_starts[0] + num_bins_before*bin_size)
            elif t_index_s-t_index_e == 2:
                new_t = round(bin_starts[0] + (num_bins_before+1)*bin_size)
            else:
                new_t = round(bin_starts[0] + num_bins_before*bin_size + 
                              bin_size*(t-bin_starts[num_bins_before])/(bin_ends[num_bins_before]-bin_starts[num_bins_before]))
            new_t_list.append(new_t)
        
    else:
        for t in t_list:
            t_index = bisect.bisect_left(binturns, t)
            new_t_list.append(round(binturns[0] + bin_size*(t_index-1) + 
                                    bin_size/(binturns[t_index]-binturns[t_index-1])*(t-binturns[t_index-1])))
        
    return new_t_list


# In[ ]:


def reorder(shuffled_song, pattern):
    new_song = list()
    for i in range(len(shuffled_song)):
        sorted_pattern = sorted(list(enumerate(pattern[i])), key=lambda tup: tup[1])
        newslice = [shuffled_song[i][j] for j, k in sorted_pattern]
        new_song.append(newslice)
    return(new_song)


# ## Synthetic song prediction

# In[5]:


def make_datasets_finch_synth(neural_file_name, song_file_name, motif_file_name, song_length, bin_size, num_clusters, num_lookbacks, bird_id, model_name,
                        start_extend_bins=0, end_extend_bins=0, specify_subdir=None,
                        n_mel_freq_components = 64, start_freq = 200, end_freq = 15000, slice_shuffle_pattern=None, 
                        warp_method=None):
    '''
    neural_file_name: kwik file that contains neural data
    song_file_name: kwik file that contains motif starting time and recordings
    song_length: length of motif
    bin_size: how many datapoints to combine into a bin
    num_clusters: how many clusters of neurons are recorded
    start_extend_bins: at the beginning of the motif, how many bins do you want to shift. Shifting the start to the left
                        is negative, and shifting to the right is positive. Default to be 0
    end_extend_bins: same as start_extend_bins, how many bins to shift at the end of motif. Shifting left is negative and
                        right is positive. Default to be 0.
    warp_method: None of not warping, 'anderson' if using warping algorithm published by anderson et al
    '''
    if slice_shuffle_pattern and (start_extend_bins or end_extend_bins):
        raise ValueError('Slice shuffling algorithm does not support extend bins now.')
    print('Current warping method is: '+str(warp_method))
    
    bird_folder_data, bird_folder_save, repos_folder, results_folder = locate_folders(bird_id, model_name)
    save_name = 'mel_%02d_%d_%d_lstm.p' %(num_lookbacks, start_extend_bins, end_extend_bins)
    
    start_extend_bins, end_extend_bins = cal_extend_size(bin_size, start_extend_bins, end_extend_bins)
    num_bins = cal_num_bins(song_length, bin_size, start_extend_bins, end_extend_bins)
    subdirs = which_dir(bird_folder_data, specify_subdir)
    
    '''
    all_batches = list()
    for _ in range(num_bins):
        all_batches.append(list())
    '''
    mel_neural_comb=list()
    num_bins_list = list()
    
    slice_shuffle_pattern, automatic_shuffle = make_shuffle_pattern(slice_shuffle_pattern, num_bins, n_mel_freq_components)
    
    for directory in subdirs:
        kwik_file = os.path.join(directory, neural_file_name)
        song_file = os.path.join(directory, song_file_name)
        
        try:
            trial1 = h5py.File(kwik_file,'r')['/channel_groups/0/spikes/time_samples'].value
        except IOError:
            print('--->Skipping '+directory+' because it doesnt contain spiking kwik files.')
            continue
            
        if "sleep" in directory:
            print('--->Skipping '+directory+' because it is a sleep recording.')
            continue
        
        recording = directory[directory.index('day-'):]
        recording_folder_save = os.path.join(bird_folder_save, recording)
        if not os.path.exists(recording_folder_save):
            os.makedirs(recording_folder_save)
        
        time_samples_each_rec, cluster_each_rec = separate_recs(directory, neural_file_name, bird_id, model_name)
            
        mel_neural_comb_dir  = calculate_neural_mel_synth(directory, time_samples_each_rec, cluster_each_rec,
                                                     recording_folder_save, slice_shuffle_pattern,
                                                     neural_file_name, song_file_name, bird_id, model_name, num_clusters, bin_size,
                                                     num_lookbacks, n_mel_freq_components, start_freq, end_freq, start_extend_bins, 
                                                     end_extend_bins, motif_file_name, warp_method = warp_method)
        num_bins_dir = [len(mel_neural) for mel_neural in mel_neural_comb_dir]
        mel_neural_comb += mel_neural_comb_dir
        num_bins_list+=num_bins_dir
        
        #mel_neural_comb is a collection of lists, each inidividual list is a song, contains a colletion of neural-mel 
        #tuples within a song
        
        print('Finished.')
        
    #mel_neural_comb = song_based_to_all(mel_neural_comb, num_bins_list)
         
    pickle.dump(mel_neural_comb, open(os.path.join(bird_folder_save, save_name),'wb'))
    
    if automatic_shuffle:
        return mel_neural_comb, slice_shuffle_pattern
    else:
        return mel_neural_comb


# In[ ]:


def calculate_neural_mel_synth(directory, time_samples_each_rec, cluster_each_rec, 
                         recording_folder_save, slice_shuffle_pattern,
                         neural_file_name, song_file_name, bird_id, model_name, num_clusters, bin_size, 
                         num_lookbacks, n_mel_freq_components, start_freq, end_freq, start_extend_bins, 
                         end_extend_bins, motif_file_name, mode='neuro2spec', labels=None, flatten_inputs = False, plot_raster = False,
                        warp_method = None):
    if song_file_name.endswith('kwik') or song_file_name.endswith('kwe'):
        song_file_type = 'kwik'
    else:
        song_file_type = 'pickle'
    song_file = os.path.join(directory, song_file_name)
    
    fig_folder_save = os.path.join(recording_folder_save, 'mel', '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now()))
    
    if not os.path.exists(fig_folder_save):
        os.makedirs(fig_folder_save)
        
    mel_neural_comb = list()
    num_bins_list = list()
    
    motif = pickle.load(open(motif_file_name, 'r'))
    
    if song_file_type=='kwik':
        motif_starts = h5py.File(song_file, 'r')['event_types/singing/motiff_1/time_samples'].value  #start time of each motif
        motif_recs = h5py.File(song_file, 'r')['event_types/singing/motiff_1/recording'].value  #recording number of each motif
        songtuples = zip(motif_recs, motif_starts)
        
        positions = [[] for i in range(num_clusters)]
        spec_list = list()
        wav_list = list()
        count = 0

        for rec, ideal_start in songtuples:
            mel_neural_comb_song = list()
            
            this_motif = motif.loc[(motif.start==ideal_start) & (motif.rec==rec)]
            if not len(this_motif):
                print('Can''t find %d in rec %d' %(ideal_start, rec))
                continue
            
            wav_data = this_motif.syn_song.iloc[0]
            
            #if wav_data[0]==0 & wav_data[1]==0:
            #    print('%d in rec %d does not contain a valid waveform' %(ideal_start, rec))
            #    continue
            
            this_song_length = len(wav_data)-cal_fft_size(bin_size)
            this_num_bins = cal_num_bins(this_song_length, bin_size, start_extend_bins, end_extend_bins)
            this_song_length = (this_num_bins+2+start_extend_bins-end_extend_bins)*bin_size
            extended_song_length = this_song_length+cal_fft_size(bin_size)
            
            extended_song_length, neural_song_length = cal_modified_length(this_song_length, bin_size, start_extend_bins, 
                                                                           end_extend_bins, num_lookbacks)
            wav_data = wav_data[:extended_song_length]
            
            if mode=='neuro2spec':
                neural_ideal_start = ideal_start+bin_size*(start_extend_bins+1-num_lookbacks)
                neural_ideal_end = neural_ideal_start+neural_song_length

                song_ideal_start = ideal_start+bin_size*start_extend_bins
                song_ideal_end = song_ideal_start+extended_song_length
                song_ideal_end = song_ideal_end+int((16-num_lookbacks)%16)*bin_size
                
            elif mode=='spec2neuro':
                
                song_ideal_start = ideal_start+bin_size*(start_extend_bins-num_lookbacks)
                song_ideal_end = song_ideal_start+extended_song_length
                
                song_ideal_end = song_ideal_end+int((16-num_lookbacks)%16)*bin_size
                
                neural_ideal_start = ideal_start+bin_size*(start_extend_bins+1)
                neural_ideal_end = neural_ideal_start+neural_song_length

                
            #process wav file into mel spec
            
            wav_rate=30000
            
            mel_list = make_specs(wav_data, rec, song_ideal_start, song_ideal_end, fig_folder_save, bin_size, n_mel_freq_components, 
                                  start_freq, end_freq)
            
            if np.any(np.isnan(np.array(mel_list))):
                print('%d in rec %d does not contain a valid waveform' %(ideal_start, rec))
                continue
            
            wav_list.append(wav_data)
            
            current_t_list = time_samples_each_rec[rec] #extract neural activities for current motif
            current_cluster_list = cluster_each_rec[rec]  #extract cluster info for current motif
            
            #add here warps
            if warp_method == 'anderson':
                if not count:
                    temp_list = None
                    mel_list, w_binstarts, w_binends = anderson_warp(mel_list, temp_list, fig_folder_save, bin_size, num_lookbacks, 
                                                                        song_ideal_start, count)
                    temp_list = mel_list[:]
                else:
                    mel_list, w_binstarts, w_binends = anderson_warp(mel_list, temp_list, fig_folder_save, bin_size, num_lookbacks, 
                                                                        song_ideal_start, count)
                w_binturns = [w_binstarts, w_binends]
                counts_list = spike_counter_warp(current_t_list, current_cluster_list, w_binturns, rec,
                                              fig_folder_save, count, 
                                              num_clusters, num_bins, num_lookbacks, bin_size, plot_raster=plot_raster)
            else:
                counts_list = spike_counter(current_t_list, current_cluster_list, neural_ideal_start, neural_ideal_end, song_ideal_start, rec,
                                              fig_folder_save,
                                              num_clusters, bin_size, this_num_bins, num_lookbacks)
            
            if plot_raster:
                if warp_method == 'anderson':
                    start_index = bisect.bisect_left(current_t_list, song_ideal_start)
                    end_index = bisect.bisect_left(current_t_list, song_ideal_end)
                    
                    t_list = current_t_list[start_index:end_index]
                    new_t_list = warp_spike_train(w_binturns, bin_size, t_list, warp_method=warp_method)

                    this_time = [int(t-w_binstarts[0]) for t in new_t_list]
                    this_cluster = current_cluster_list[start_index:end_index]
                    for c in range(num_clusters):
                        positions[c].append(np.array([this_time[i] for i, x in enumerate(this_cluster) if x == c]))
                    
                    np.save(os.path.join(fig_folder_save, 'positions.npy'), positions)
                        
                else:
                    positions = prep_raster(current_t_list, current_cluster_list, song_ideal_start, extended_song_length, positions,
                                            num_clusters, fig_folder_save)
                
            for i in range(this_num_bins):
                if mode=='neuro2spec':
                    inputs = np.array(counts_list[i:i+num_lookbacks])
                    if flatten_inputs:
                        inputs=inputs.flatten()
                    outputs = np.array(mel_list[i])
                    
                    #if shuffle, then update mel_list
                    if slice_shuffle_pattern:
                        outputs = outputs[np.array(slice_shuffle_pattern[i])]
                        mel_list[i] = outputs[:]
                    mel_neural_tuple = (inputs, outputs)
                    
                    mel_neural_comb_song.append(mel_neural_tuple)
                elif mode=='spec2neuro':
                    inputs = np.array(mel_list[i:i+num_lookbacks])
                    if flatten_inputs:
                        inputs=inputs.flatten()
                    mel_neural_tuple = (inputs, counts_list[i])
                    mel_neural_comb_song.append(mel_neural_tuple)
               
            spec_list.append(np.array(mel_list))
            
            if counts_list[0].shape[0] != num_clusters:
                raise ValueError('check cluster histogram')
            mel_neural_comb.append(mel_neural_comb_song)
            count+=1
        if plot_raster:
            raster_plotter(positions, num_clusters, fig_folder_save)
        spec_file = os.path.join(fig_folder_save, 'specs.npy')
        np.save(spec_file, np.array(spec_list))
        wav_file = os.path.join(fig_folder_save, 'wavs.npy')
        np.save(wav_file, np.array(wav_list))
            
    else:
        try:
            pickle_data = pd.read_pickle(song_file)
        except:
            song_file = os.path.join('/mnt/cube/earneodo/bci_zf/proc_data/', bird_id, song_file_name)
        pickle_data = pd.read_pickle(song_file)
        indeces = [i for i, j in enumerate(pickle_data['recon_folder']) if directory in j]
        indeces = [i for i, j in zip(indeces, pickle_data['syllable_labels'][indeces]) if j in labels]
        total_bins = 0
            
        for ind in indeces:
            mel_neural_comb_song = list()
            wav_file = pickle_data['recon_folder'][ind]
            rec = int(wav_file[wav_file.index('rec_')+4:wav_file.index('rec_')+7])
            ori_song_length = int(pickle_data['recon_length'][ind]*30000)
            song_length, neural_length = cal_modified_length(ori_song_length, bin_size, start_extend_bins=0, end_extend_bins=0, 
                                                                 num_lookbacks=num_lookbacks)
            num_bins = int(np.ceil(ori_song_length/bin_size-2))

            total_bins+=num_bins

            song_ideal_start = int(pickle_data['recon_t_rel_wav'][ind]*30000)
            song_ideal_end = song_ideal_start+song_length
            song_ideal_end = song_ideal_end+int((16-num_lookbacks)%16)*bin_size

            neural_ideal_start = song_ideal_start+bin_size*(1-num_lookbacks)
            neural_ideal_end = neural_ideal_start+neural_song_length

            wav_rate, wav_data = wavfile.read(wav_file)
            wav_data = wav_data[song_ideal_start:song_ideal_end]
            
            mel_list = make_specs(wav_data, wav_rate, rec, song_ideal_start, song_ideal_end, fig_folder_save, bin_size, n_mel_freq_components, 
                                  start_freq, end_freq)

            current_t_list = time_samples_each_rec[rec] #extract neural activities for current motif
            current_cluster_list = cluster_each_rec[rec]  #extract cluster info for current motif

            counts_list = spike_counter(current_t_list, current_cluster_list, neural_ideal_start, neural_ideal_end, song_ideal_start, rec,
                                          fig_folder_save,
                                          num_clusters, bin_size, num_bins, num_lookbacks)

            for i in range(num_bins):
                mel_neural_tuple = (np.array(counts_list[i:i+num_lookbacks]), np.array(mel_list[i]))
                mel_neural_comb_song.append(mel_neural_tuple)
            
            mel_neural_comb.append(mel_neural_comb_song)
                
    return mel_neural_comb

