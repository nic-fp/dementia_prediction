import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import fire
#For this project, I am using a CNN so I need fixed length inputs
#These are parameters I will tune


class CNN(torch.nn.Module):
    def __init__(self, in_channels,  hidden_layer_size = 64, num_hidden_layers = 4):
        super().__init__()
        self.layer_1 = torch.nn.Conv2d(in_channels, hidden_layer_size, kernel_size=3, stride = 1, padding =1)
        self.relu_1 = torch.nn.ReLU()
        self.layer_final = torch.nn.Conv2d(hidden_layer_size, 1, kernel_size = 1)

        layers = []
        layers.append(self.layer_1)
        layers.append(self.relu_1)

        for i in range(num_hidden_layers):
            layers.append(torch.nn.Conv2d(hidden_layer_size, hidden_layer_size, kernel_size=3, stride=1, padding=1))
            layers.append(torch.nn.BatchNorm2d(hidden_layer_size))
            layers.append(torch.nn.ReLU())
            #add max pooling every other layer. This is tunable based on how aggressivley to downsample
            if (i+1)%2 ==0 :
                layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(self.layer_final)

        #Sequential is used here bc our model is pretty simple so it will handle the forward pass
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        preds = self.network(x).mean(dim=2).mean(dim=2)
        return preds

    def predict(self, x):
        return (torch.sigmoid(self(x)) >= 0.5).long().squeeze(1)

    
class SpectrogramDataset(Dataset):
    #initialization takes df containing label column and path column
    def __init__(self, df, duration, sample_rate):
        self.df = df.reset_index(drop=True)
        self.df['final_label'] = (self.df['label'] == 'dementia').astype(int)
        self.duration = duration
        self.sample_rate = sample_rate
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = row['google_path']
        audio = process_audio_file(path, self.duration, self.sample_rate)

        label = torch.tensor(row['final_label'].astype('float32')).unsqueeze(0) #unsqueeze for batchsize
        

        return audio, label


def process_audio_file(path, duration, sample_rate):
    """
    This function takes in a path. 
    It loads the audio file at the SAMPLE_RATE, then either pads or truncates based on the duration  
    Pads with 0's

    It then generates the MEL spectrogram, normalizes and converts to a tesor
    Note that it normalizes only the non-padded audio
    Returns a torch tensor
    """
    audio, sr = librosa.load(
        path,
        sr=sample_rate,
        mono=True
    )
    
    if len(audio) > sample_rate*duration:
        audio = audio[:sample_rate*duration]
    #higher hop length = lower dimensionality
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, hop_length = 1024)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize using only the real audio. Will pad later
    mean = mel_db.mean()
    std = mel_db.std()
    mel_db = (mel_db - mean) / (std + 1e-6)

    target_frames = 1 + (sample_rate*duration) // 1024 #1024 = hop length from mel spectorgram

    # Calculate required padding
    pad_width = max(0, target_frames - mel_db.shape[1])

    # Pad along time dimension. 0 here means we are actually padding with the mean. 
    #I think this is better than padding with 0;s then normalzing
    mel_db = np.pad(
        mel_db,
        ((0, 0), (0, pad_width)),
        mode="constant",
        constant_values=0
    )
    return torch.tensor(
        mel_db,
        dtype=torch.float32
    ).unsqueeze(0)

def save_model(model,duration):
    output_path = f"results/cnn/model/cnn_{duration}.th"
    torch.save(model.state_dict(), output_path) 

def train(train_loader, val_loader, num_epochs,duration,learning_rate): 
    IN_CHANNEL = 1
    HIDDEN_LAYER_SIZE = 64
    NUM_HIDDEN_LAYERS = 4

    writer = SummaryWriter("logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(in_channels=IN_CHANNEL, hidden_layer_size=HIDDEN_LAYER_SIZE, num_hidden_layers=NUM_HIDDEN_LAYERS)

    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    loss_fn =  torch.nn.BCEWithLogitsLoss()
    best = 0
    
    for epoch in range(num_epochs):
        print(f"{epoch=}")
        model.train()
        train_acc = []
        losses = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            

            loss = loss_fn(output, label)
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).float()

            accuracy = (preds == label).float().mean().cpu()
            train_acc.append(accuracy.item())
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()  
            optim.step()
            
        print(f"Train acc: {np.mean(train_acc):.4f}") 
        print(f"Train loss: {np.mean(losses):.4f}")
        model.eval()
        val_accuracy = []
        val_losses = []
        for data, label in val_loader:
            data, label = data.to(device), label.to(device)
            #need this so gradients don't accumulate
            with torch.inference_mode():
                output = model(data)
            
                loss = loss_fn(output,label)
                probs = torch.sigmoid(output)
                preds = (probs > 0.5).float()
                val_acc = (preds == label).float().mean().cpu()
                val_accuracy.append(val_acc.item())
                val_losses.append(loss.item())
            
        print(f"Val acc : {np.mean(val_accuracy)}")
        print(f"Val loss : {np.mean(val_losses)}")
        if np.mean(val_accuracy) > best: 
            save_model(model,duration)
        writer.add_scalar("Loss/train", np.mean(losses), epoch)
        writer.add_scalar("Loss/validation", np.mean(val_losses), epoch)

        writer.add_scalar("Accuracy/train", np.mean(train_acc), epoch)
        writer.add_scalar("Accuracy/validation", np.mean(val_accuracy), epoch)

def overfit_model(train_loader):
    IN_CHANNEL = 1
    HIDDEN_LAYER_SIZE = 64
    NUM_HIDDEN_LAYERS = 4

    writer = SummaryWriter("logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(in_channels=IN_CHANNEL, hidden_layer_size=HIDDEN_LAYER_SIZE, num_hidden_layers=NUM_HIDDEN_LAYERS)

    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr = .001)
    loss_fn =  torch.nn.BCEWithLogitsLoss()
    data, label = next(iter(train_loader))
    data, label = data.to(device), label.to(device).float().view(-1, 1)
    model.train()
    for step in range(1000):
        output = model(data)
        loss = loss_fn(output, label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 10 == 0:
            with torch.no_grad():
                preds = (torch.sigmoid(output) > 0.5).float()
                acc = (preds == label).float().mean().item()

            print(f"Step {step}: loss={loss.item():.4f}, accuracy={acc:.3f}")
        
def main(num_epochs=30, learning_rate=.001,duration=45, sample_rate=44100, batch_size=16,overfit=False):
    # _2 has the google path for training in colab
    train_data = pd.read_csv('/content/drive/My Drive/data/train_dm_transcribed_2.csv')
    test_data = pd.read_csv('/content/drive/My Drive/data/valid_dm_transcribed_2.csv')
    train_dataset = SpectrogramDataset(train_data, duration,sample_rate)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    #Split into val and train for eval while training
    train_dataset, val_dataset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=True)
    if overfit:
        overfit_model(train_loader)
    else:
        train(train_loader, val_loader, num_epochs,duration,learning_rate)



    test_dataset = SpectrogramDataset(test_data,duration, sample_rate)
    #print(process_audio_file('data/dementia/Robin Williams/RobinWilliams_0.wav').shape)
    #process_audio_file('data/dementia/Terry Jones/terryjones_5.wav')
if __name__ == "__main__":
    fire.Fire(main)
