import librosa
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(123)

"""
Pandas structure
label 0-n
path absolute path
"""


def split_data(df: pd.DataFrame,
               train_split: float = 0.7,
               seed: int = 1234):
    train, aux = train_test_split(df, train_size=train_split, random_state=seed)
    val, test = train_test_split(aux, train_size=0.5, random_state=seed)
    return train, val, test


def load_audio_duration_fixed(df: pd.DataFrame,
                              duration: float,
                              sample_rate: int,
                              offset: float = 0.5, ):
    audio_list = []
    for filepath in df['filepath']:
        audio, sample_rate = librosa.load(filepath, duration=duration, offset=offset, sr=sample_rate)
        signal = np.zeros((int(sample_rate * 3, )))
        signal[:len(audio)] = audio
        audio_list.append(signal)

    return np.stack(audio_list, axis=0)


def additive_white_gaussian_noise(signal, num_bits=16, augmented_num=2, snr_low=15, snr_high=30):
    # AdditiveWhiteGaussianNoise
    signal_len = len(signal)
    noise = np.random.normal(size=(augmented_num, signal_len), )
    norm_constant = 2.0 ** (num_bits - 1)
    signal_norm = signal / norm_constant
    noise_norm = noise / norm_constant
    s_power = np.sum(signal_norm ** 2) / signal_len
    n_power = np.sum(noise_norm ** 2, axis=1) / signal_len
    target_snr = np.random.randint(snr_low, snr_high)
    K = np.sqrt((s_power / n_power) * 10 ** (- target_snr / 10))
    K = np.ones((signal_len, augmented_num)) * K
    return signal + K.T * noise


def add_adwgn(data_array, labels):
    aug_signals = []
    aug_labels = []
    for audio_signal, label in zip(data_array, labels):
        augmented_signals = additive_white_gaussian_noise(audio_signal)
        for j in range(augmented_signals.shape[0]):
            aug_labels.append(label)
            aug_signals.append(augmented_signals[j, :])
    aug_signals = np.stack(aug_signals, axis=0)
    X_aug = np.concatenate([data_array, aug_signals], axis=0)
    aug_labels = np.stack(aug_labels, axis=0)
    Y_aug = np.concatenate([labels, aug_labels])
    return X_aug, Y_aug


def standard_scaler(X_array):
    X_array = np.expand_dims(X_array, 1)
    scaler = StandardScaler()
    b, c, h, w = X_array.shape
    X_array = np.reshape(X_array, newshape=(b, -1))
    X_array = scaler.fit_transform(X_array)
    X_array = np.reshape(X_array, newshape=(b, c, h, w))
    return X_array


def get_mel_spectrogram(audio, sample_rate=16000, n_fft=1024, win_legth=512, n_mels=128):
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                              sr=sample_rate,
                                              n_fft=n_fft,
                                              win_length=win_legth,
                                              window='hamming',
                                              hop_length=256,
                                              n_mels=n_mels,
                                              fmax=sample_rate / 2
                                              )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def get_mel_from_data(X, sample_rate=16000):
    mel_list = [get_mel_spectrogram(audio, sample_rate=sample_rate) for audio in X]
    return np.stack(mel_list, axis=0)
