''' First task is going to be feature extraction from the audio files '''
import librosa
import numpy as np
import scipy
import os
import pickle
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

def describe_freq(freqs):
    mean = np.mean(freqs)
    std = np.std(freqs) 
    maxv = np.amax(freqs) 
    minv = np.amin(freqs) 
    median = np.median(freqs)
    skew = scipy.stats.skew(freqs)
    kurt = scipy.stats.kurtosis(freqs)
    q1 = np.quantile(freqs, 0.25)
    q3 = np.quantile(freqs, 0.75)
    mode = scipy.stats.mode(freqs)[0][0]
    iqr = scipy.stats.iqr(freqs)
    
    return [mean, std, maxv, minv, median, skew, kurt, q1, q3, mode, iqr]


def energy(x):
    return np.sum(x**2)


def featurize(wavfile):
    hop_length = 512
    x, sr = librosa.load(wavfile)
    freqs = np.fft.fftfreq(x.size)
    # statistical features
    statistical_features = describe_freq(freqs)
    # Mel frequency Cepstral coefficients (MFCC)
    mfcc = librosa.feature.mfcc(x, sr=sr, hop_length = hop_length, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_amin = np.amin(mfcc, axis=1)
    mfcc_amax = np.amax(mfcc, axis=1)

    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
    mfcc_delta_std = np.std(mfcc_delta, axis=1)
    mfcc_delta_amin = np.amin(mfcc_delta, axis=1)
    mfcc_delta_amax = np.amax(mfcc_delta, axis=1)
    mfcc_features = np.array([mfcc_mean, mfcc_std, mfcc_amin, mfcc_amax,
                             mfcc_delta_mean, mfcc_delta_std, mfcc_delta_amin, mfcc_delta_amax]).reshape(-1)

    final_features = np.append(np.array(statistical_features), mfcc_features)
    return final_features

if __name__=="__main__":
    male_audio_files = os.listdir("../../data/VoxCeleb/males")
    female_audio_files = os.listdir("../../data/VoxCeleb/females")
    males = []
    females = []
    
    print("Male")
    for file in tqdm(male_audio_files):
        mfcc_features = featurize("../../data/VoxCeleb/males/"+file)
        males.append(mfcc_features)

    print("Female")
    for file in tqdm(female_audio_files):
        mfcc_features = featurize("../../data/VoxCeleb/females/"+file)
        females.append(mfcc_features)

    with open('../../data/VoxCeleb/male_data.pkl', 'wb') as handle:
        pickle.dump(males, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open("../../data/VoxCeleb/female_data.pkl", 'wb') as handle:
        pickle.dump(females, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
