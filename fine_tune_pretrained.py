from datasets import load_dataset, Audio, Value
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
)
import fire
import torch
import numpy as np
from sklearn.metrics import accuracy_score



MODELS_DICT = {
    'wav2vec':{'ckpt':'facebook/wav2vec2-base','sr':16000},
    'hubert':{'ckpt':'facebook/hubert-base-ls960','sr':16000},
    'whisper':{'ckpt':'openai/whisper-base','sr':16000}
}

class CustomCollator:
    """
    This class will be passed to the trainer as the data collator
    Pads our audio files PER batch (rather than overall)
    """
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    #override what happens when we call the collator
    def __call__(self, features):
        
        audio_features = [i['audio']['array'] for i in features]
        labels = [i['label'] for i in features]
        #for each INDIVIDUAL batch we are going to pad to be longest in batch
        #Note how this differs from CNN architecture where we padded all to fixed length
        batch = self.feature_extractor(audio_features, sampling_rate=self.feature_extractor.sampling_rate, 
                                       padding=True, return_tensors='pt')
        batch['labels'] = torch.tensor(labels,dtype=torch.long)
        return batch

def get_dataset(model_name, data_path):
    dataset = load_dataset("csv",data_files=data_path)
    dataset = dataset["train"]
    #weird bug where it loads the path col as LongString instead. So you have to cast to string then cast to audio otherwise errors
    dataset = dataset.cast_column("path", Value("string"))
    dataset = dataset.cast_column("path", Audio(sampling_rate=16_000))
    #Saw this in docs for conventions
    dataset = dataset.rename_column("path", "audio")
    return dataset

def load_model(model_name):
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODELS_DICT[model_name]['ckpt'])
    model = AutoModelForAudioClassification.from_pretrained(
        MODELS_DICT[model_name]['ckpt'],
        num_labels=2,
        
        #ignore_mismatched_sizes=True,
        #these are used by the trainer so I don't have to encode!
        label2id={'nodementia': 0, 'dementia': 1},
        id2label={0: 'nodementia', 1: 'dementia'}
    )
    return model, feature_extractor

def train_model(model,feature_extractor, train_dataset, model_name):

    split_dataset = train_dataset.train_test_split(test_size=0.2, seed=42)

    # Extract your individual datasets
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    def compute_metrics(eval_pred):
        probs, labels = eval_pred
        predictions = np.argmax(probs, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}
    
    training_args = TrainingArguments(
        output_dir=f'results/finetune_{model_name}', 
        logging_strategy="epoch",
        eval_strategy = "epoch",
        report_to='tensorboard', 
        gradient_checkpointing=True, 
        learning_rate=.001, 
        num_train_epochs=5, 
        per_device_train_batch_size=4,
        remove_unused_columns=False,

    )
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CustomCollator(feature_extractor)
    )
    trainer.train()
    trainer.save_model(f'results/finetune_{model_name}')

def main(model_name, in_colab = False):
    """
    Allowed values for model_name are found in the keys in MODELS_DICT
    """
    train_data = get_dataset(model_name, f'data/train_dm_transcribed{'_2' if in_colab else ''}.csv')
    model, feature_extractor = load_model(model_name)
    train_model(model, feature_extractor, train_data, model_name)


if __name__ == "__main__":
    fire.Fire(main)
