import librosa
import numpy as np

def extract_mfcc(y, sr, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.hstack((mfcc_mean, mfcc_std))  # 80-length vector
def extract_additional_features(y, sr):
    features = {}

    features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))
    features['rms'] = np.mean(librosa.feature.rms(y=y))
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = np.mean(chroma)
    features['chroma_std'] = np.std(chroma)

    return features
def extract_full_feature_vector(y, sr):
    mfcc_feat = extract_mfcc(y, sr)
    add_feats = extract_additional_features(y, sr)
    combined = np.concatenate([mfcc_feat, np.array(list(add_feats.values()))])
    return combined, list(add_feats.keys())

