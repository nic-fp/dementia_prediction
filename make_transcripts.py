from faster_whisper import WhisperModel
import pandas as pd
from faster_whisper import WhisperModel


def make_transcript(row, model, language=None):
    # Initialize model
    
    print(f'Transcribing {row['file']}')
    # Transcribe audio NOTE!!! YOu have to have av version at <19 for this to work
    if language is None:
        segments, info = model.transcribe(
            row['path'],
            #language='en',  #experiment with commenting this out to see if worse transcriptions ==> Predictive of dementia
            beam_size=5 #beamsize is how many // paths it looks at at once and then chooses best at the end
        )
    else:
        segments, info = model.transcribe(
                    row['path'],
                    language='en',  #experiment with commenting this out to see if worse transcriptions ==> Predictive of dementia
                    beam_size=5 #beamsize is how many // paths it looks at at once and then chooses best at the end
                )
    print('done')
    # Combine transcript segments
    transcript = ""

    for segment in segments:
        transcript += segment.text + " "

    return pd.Series({"language":info.language, 'transcript':transcript})


def main():
    #initialize the model one time
    model = WhisperModel(
            "base",
            device="cuda",
            compute_type="int8"
        )
    print('Model loaded')
    #load data and cleean the file path
    train_data = pd.read_csv('data/train_dm.csv',sep='\t')
    train_data = train_data.replace('/content/drive/My Drive/type3/', '', regex=True)
    val_data = pd.read_csv('data/valid_dm.csv',sep='\t')
    val_data = val_data.replace('/content/drive/My Drive/type3/', '', regex=True)
    #get transcript using whisper model
    print('Starting transcription for training data')
    train_data[['language', 'transcript']] = train_data.apply(make_transcript,model=model, axis=1, result_type='expand')
    print('Starting transcripting for val data')
    val_data[['language', 'transcript']] =val_data.apply(make_transcript, model=model,axis=1, result_type='expand')
    print("Done transcribing.\nWriting to csv's")
    #write to new csv (don't want to affect the original)
    train_data.to_csv('data/train_dm_transcribed_any_lang.csv', index=False)
    val_data.to_csv('data/valid_dm_transcribed_any_lang.csv', index=False)


    print('Starting transcription for training data')
    train_data[['language', 'transcript']] = train_data.apply(make_transcript,model=model,language='en', axis=1, result_type='expand')
    print('Starting transcripting for val data')
    val_data[['language', 'transcript']] =val_data.apply(make_transcript, model=model, language='en',axis=1, result_type='expand')
    print("Done transcribing.\nWriting to csv's")
    #write to new csv (don't want to affect the original)
    train_data.to_csv('data/train_dm_transcribed.csv', index=False)
    val_data.to_csv('data/valid_dm_transcribed.csv', index=False)


if __name__ == "__main__":
    main()
