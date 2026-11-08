import pandas as pd
import librosa
import numpy as np
import textstat
import nltk
import re   

def get_audio_columns(filepath):
    """
    This function takes in a filepath, loads the audio files for each and adds summary data to the respective rows
    All audio features will start with audio_ so that they can be used separately later
    Features extracted:
    Duration in seconds
    RMS median, mean, std, min, max
    ZCR mean, std  
    Spectral Centroid mean, std   
    Spectral Bandwith mean, std  
    Mfcc 1-13 mean, std
    Returns dictionary containing all of the audio features
    """
    y, sample_rate = librosa.load(filepath, sr=None)
    features = {}
    features['audio_duration'] = len(y)/sample_rate
    #for each frame, it computes the rms value. So I will extract the rms, then aggregate the frames
    #recall rms is basically volume
    rms = librosa.feature.rms(y=y)[0]
    features["audio_rms_median"] = np.median(rms)
    features["audio_rms_mean"] = np.mean(rms)
    features["audio_rms_std"] = np.std(rms)
    features["audio_rms_min"] = np.min(rms)
    features["audio_rms_max"] = np.max(rms)

    #similarly for Zero cross rate
    #recall zcr is how many time freq of signal crosses zero. Higher is higher pitched signal
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    features["audio_zcr_mean"] = np.mean(zcr)
    features["audio_zcr_std"] = np.std(zcr)

    #centroid is basically center of mass 
    #bandwith is more so the variance around the centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sample_rate)[0]
    

    features["audio_spectral_centroid_mean"] = np.mean(centroid)
    features["audio_spectral_centroid_std"] = np.std(centroid)

    features["audio_spectral_bandwidth_mean"] = np.mean(bandwidth)
    features["audio_spectral_bandwidth_std"] = np.std(bandwidth)

    # Mel-frequency cepstral coefficients (MFCCs)
    #per some research, 13 is a pretty standard number to use
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sample_rate,
        n_mfcc=13
    )
    for i in range(len(mfccs)):
        features[f"audio_mfcc_{i+1}_mean"] = np.mean(mfccs[i])
        features[f"audio_mfcc_{i+1}_std"] = np.std(mfccs[i])
    return features

def get_text_columns(transcript, duration):
    """
    This function takes in a transcript and the duration (calculated in get_audio_columns), and calculates lingustic summary data
    All features will start with text_ so that they can be used separately later
    Features extracted:
    # of words
    Words per second 
    # of sentences
    Sentences per second   
    Avg word length
    SMOG Index 
    Flesch Reading Ease
    Returns dictionary containing all of the text features
    """
    features = {}
    individual_words = re.findall(r'\b\w+\b', transcript)
    word_count = len(individual_words)
    features['text_word_count'] = word_count
    features['text_words_per_second'] = word_count / duration 


    sentence_count = len(nltk.tokenize.sent_tokenize(transcript))
    features['text_sentence_count'] = sentence_count  
    features['text_sentences_per_second'] = sentence_count / duration 


    chars_per_word = sum(len(word) for word in individual_words) / len(individual_words)
    features['text_avg_word_length'] = chars_per_word


    flesch_reading_ease = textstat.flesch_reading_ease(transcript)
    features['text_flesch_reading_ease'] = flesch_reading_ease  

    # SMOG Index (Simple Measure of Gobbledygook)
    smog_index = textstat.smog_index(transcript)
    features['text_smog_index'] = smog_index
    return features


def create_data(df):
    """
    This is a driver function that takes a datafame 
    Runs both get audio columns and get text columns  
    Returns dataframe
    """
    audio_columns = df['path'].apply(get_audio_columns)
    # Convert dictionaries into columns
    audio_features_df = pd.DataFrame(audio_columns.tolist(), index=df.index)
    
    # Combine with original dataframe
    df = pd.concat([df, audio_features_df], axis=1)
    text_columns = df.apply(
        lambda row: get_text_columns(row['transcript'], row['audio_duration']),
        axis=1
    )
    # Convert dictionaries into columns
    text_features_df = pd.DataFrame(text_columns.tolist(), index=df.index)
    df = pd.concat([df, text_features_df], axis=1)
    return df

def main():
    #these are used to get sentence count. Uncomment if you havent downloaded previously
    #nltk.download('punkt')
    #nltk.download('punkt_tab')
    train_data = pd.read_csv('data/train_dm_transcribed.csv')
    print(train_data.head())
    train_data = create_data(train_data)
    

    train_data.to_csv('data/tabular/train_data.csv')
    print(train_data.head())


    val_data = pd.read_csv('data/valid_dm_transcribed.csv')
    print(val_data.head())
    val_data = create_data(val_data)
    

    val_data.to_csv('data/tabular/val_data.csv')
    print(val_data.head())

if __name__ == "__main__":
    main()
