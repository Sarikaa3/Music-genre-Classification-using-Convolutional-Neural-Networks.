
GENRES      = [ 'blues',
                'classical',
                'country',
                'disco',
                'hiphop',
                'jazz',
                'metal',
                'pop',
                'reggae',
                'rock'  ]


DATAPATH        =   '../../genres/'
RAW_DATAPATH    =   '../utils/raw_data.pkl'
SET_DATAPATH    =   '../utils/set.pkl'
MODELPATH       =   '../model/net.pt'


import pandas as pd
import numpy as np
import pickle
import os

from librosa.core import load
from librosa.feature import melspectrogram
from librosa import power_to_db

class Data():

    def __init__(self, genres, datapath):
        self.raw_data   = None
        self.GENRES     = genres
        self.DATAPATH   = datapath

    def make_raw_data(self):
        records = list()
        for i, genre in enumerate(self.GENRES):
            GENREPATH = self.DATAPATH + genre + '/'
            for j, track in enumerate(os.listdir(GENREPATH)):
                TRACKPATH   = GENREPATH + track
                print("%d.%s\t\t%s (%d)" % (i + 1, genre, TRACKPATH, j + 1))
                y, sr       = load(TRACKPATH, mono=True)
                S           = melspectrogram(y, sr).T 
                S           = S[:-1 * (S.shape[0] % 128)]
                
                num_chunk   = S.shape[0] / 128
                data_chunks = np.split(S, num_chunk)
                data_chunks = [(data, genre) for data in data_chunks]
                records.append(data_chunks)

        records = [data for record in records for data in record]
        self.raw_data = pd.DataFrame.from_records(records, columns=['spectrogram', 'genre'])
        return

    def save(self):
        with open(RAW_DATAPATH, 'wb') as outfile:
            pickle.dump(self.raw_data, outfile, pickle.HIGHEST_PROTOCOL)
        return

    def load(self):
        with open(RAW_DATAPATH, 'rb') as infile:
            self.raw_data   = pickle.load(infile)
        return


np.random.seed(0)
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

class Set():

    def __init__(self, data):
        self.train_set  = None
        self.valid_set  = None
        self.test_set   = None
        self.data       = data
        self.le         = LabelEncoder().fit(self.data.GENRES)

    def make_dataset(self):
        df  = self.data.raw_data.copy()
        df  = shuffle(df)

        train_records, valid_records, test_records = list(), list(), list()
        for i, genre in enumerate(self.data.GENRES):
            genre_df    = df[df['genre'] == genre]
            train_records.append(genre_df.iloc[:700].values)
            valid_records.append(genre_df.iloc[700:900].values)
            test_records.append(genre_df.iloc[900:].values)

        train_records   = shuffle([record for genre_records in train_records    for record in genre_records])
        valid_records   = shuffle([record for genre_records in valid_records    for record in genre_records])
        test_records    = shuffle([record for genre_records in test_records     for record in genre_records])

        self.train_set  = pd.DataFrame.from_records(train_records,  columns=['spectrogram', 'genre'])
        self.valid_set  = pd.DataFrame.from_records(valid_records,  columns=['spectrogram', 'genre'])
        self.test_set   = pd.DataFrame.from_records(test_records,   columns=['spectrogram', 'genre'])
        return

    def get_train_set(self):
        x_train = np.stack(self.train_set['spectrogram'].values)
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
        y_train = np.stack(self.train_set['genre'].values)
        y_train = self.le.transform(y_train)
        print("x_train shape: ", x_train.shape)
        print("y_train shape: ", y_train.shape)
        return x_train, y_train

    def get_valid_set(self):
        x_valid = np.stack(self.valid_set['spectrogram'].values)
        x_valid = np.reshape(x_valid, (x_valid.shape[0], 1, x_valid.shape[1], x_valid.shape[2]))
        y_valid = np.stack(self.valid_set['genre'].values)
        y_valid = self.le.transform(y_valid)
        print("x_valid shape: ", x_valid.shape)
        print("y_valid shape: ", y_valid.shape)
        return x_valid, y_valid

    def get_test_set(self):
        x_test  = np.stack(self.test_set['spectrogram'].values)
        x_test  = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]))
        y_test  = np.stack(self.test_set['genre'].values)
        y_test  = self.le.transform(y_test)
        print("x_test shape : ", x_test.shape)
        print("y_test shape : ", y_test.shape)
        return x_test, y_test

    def save(self):
        with open(SET_DATAPATH, 'wb') as outfile:
            pickle.dump((self.train_set, self.valid_set, self.test_set), outfile, pickle.HIGHEST_PROTOCOL)
        return

    def load(self):
        with open(SET_DATAPATH, 'rb') as infile:
            (self.train_set, self.valid_set, self.test_set) = pickle.load(infile)
        return


import torch
torch.manual_seed(123)
from torch.nn import Module, Conv2d, MaxPool2d, Linear, Dropout, BatchNorm2d
import torch.nn.functional as F


class genreNet(Module):

    def __init__(self):
        super(genreNet, self).__init__()

        self.conv1  = Conv2d(in_channels=1,     out_channels=64,    kernel_size=3,  stride=1,   padding=1)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1    = BatchNorm2d(64)
        self.pool1  = MaxPool2d(kernel_size=2)

        self.conv2  = Conv2d(in_channels=64, out_channels=128,      kernel_size=3,  stride=1,   padding=1)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2    = BatchNorm2d(128)
        self.pool2  = MaxPool2d(kernel_size=2)

        self.conv3  = Conv2d(in_channels=128, out_channels=256,      kernel_size=3,  stride=1,   padding=1)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.bn3    = BatchNorm2d(256)
        self.pool3  = MaxPool2d(kernel_size=4)

        self.conv4  = Conv2d(in_channels=256, out_channels=512,      kernel_size=3,  stride=1,   padding=1)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        self.bn4    = BatchNorm2d(512)
        self.pool4  = MaxPool2d(kernel_size=4)

        self.fc1    = Linear(in_features=2048,  out_features=1024)
        self.drop1  = Dropout(0.5)

        self.fc2    = Linear(in_features=1024,  out_features=256)
        self.drop2  = Dropout(0.5)

        self.fc3    = Linear(in_features=256,   out_features=10)

    def forward(self, inp):
        x   = F.relu(self.bn1(self.conv1(inp)))
        x   = self.pool1(x)

        x   = F.relu(self.bn2(self.conv2(x)))
        x   = self.pool2(x)

        x   = F.relu(self.bn3(self.conv3(x)))
        x   = self.pool3(x)

        x   = F.relu(self.bn4(self.conv4(x)))
        x   = self.pool4(x)

        x   = x.view(x.size()[0], -1)
        x   = F.relu(self.fc1(x))
        x   = self.drop1(x)

        x   = F.relu(self.fc2(x))
        x   = self.drop2(x)

        x   = F.log_softmax(self.fc3(x),dim=1)
        return x
    
   
import torch
torch.manual_seed(123)
from torch.autograd import Variable


def main():
    
    data    = Data(GENRES, DATAPATH)
    data.make_raw_data()
    data.save()
    data    = Data(GENRES, DATAPATH)
    data.load()
   
    set_    = Set(data)
    set_.make_dataset()
    set_.save()
    set_ = Set(data)
    set_.load()

    x_train, y_train    = set_.get_train_set()
    x_valid, y_valid    = set_.get_valid_set()
    x_test,  y_test     = set_.get_test_set()

    TRAIN_SIZE  = len(x_train)
    VALID_SIZE  = len(x_valid)
    TEST_SIZE   = len(x_test)

    net = genreNet()
    net.cuda()

    criterion   = torch.nn.CrossEntropyLoss()
    optimizer   = torch.optim.RMSprop(net.parameters(), lr=1e-4)

    EPOCH_NUM   = 250
    BATCH_SIZE  = 16

    for epoch in range(EPOCH_NUM):
        inp_train, out_train    = Variable(torch.from_numpy(x_train)).float().cpu(), Variable(torch.from_numpy(y_train)).long().cpu()
        inp_valid, out_valid    = Variable(torch.from_numpy(x_valid)).float().cpu(), Variable(torch.from_numpy(y_valid)).long().cpu()
        print("!!!!!!!!!!!!!!")
        print(out_train)
       
        print(list(inp_train.size()),list(out_train.size()))
        train_loss = 0
        optimizer.zero_grad()  
        
        for i in range(0, TRAIN_SIZE, BATCH_SIZE):
            x_train_batch, y_train_batch = inp_train[i:i + BATCH_SIZE], out_train[i:i + BATCH_SIZE]
            print(list(x_train_batch.size()))
            pred_train_batch    = net(x_train_batch)
            print(list(pred_train_batch.size()))
            
            loss_train_batch    = criterion(pred_train_batch, y_train_batch)
            train_loss          += loss_train_batch.cpu().detach().numpy()

            loss_train_batch.backward()
        optimizer.step() 

        epoch_train_loss    = (train_loss * BATCH_SIZE) / TRAIN_SIZE
        train_sum           = 0
        for i in range(0, TRAIN_SIZE, BATCH_SIZE):
            pred_train      = net(inp_train[i:i + BATCH_SIZE])
            print("------")
            print(list(pred_train.size()))
            
            indices_train   = pred_train.max(1)[1]
            print(list(indices_train.size()))
            print(indices_train)
            train_sum       += (indices_train == out_train[i:i + BATCH_SIZE]).sum().cpu().detach().numpy()
            print(train_sum)
        train_accuracy  = train_sum / float(TRAIN_SIZE)

       
        valid_loss = 0
        for i in range(0, VALID_SIZE, BATCH_SIZE):
            x_valid_batch, y_valid_batch = inp_valid[i:i + BATCH_SIZE], out_valid[i:i + BATCH_SIZE]

            pred_valid_batch    = net(x_valid_batch)
            loss_valid_batch    = criterion(pred_valid_batch, y_valid_batch).cpu().detach().numpy()
            valid_loss          += loss_valid_batch

        epoch_valid_loss    = (valid_loss * BATCH_SIZE) / VALID_SIZE
        valid_sum           = 0
        for i in range(0, VALID_SIZE, BATCH_SIZE):
            pred_valid      = net(inp_valid[i:i + BATCH_SIZE])
            indices_valid   = pred_valid.max(1)[1]
            valid_sum       += (indices_valid == out_valid[i:i + BATCH_SIZE]).sum().cpu().detach().numpy()
        valid_accuracy  = valid_sum / float(VALID_SIZE)

        print("Epoch: %d\t\tTrain loss : %.2f\t\tValid loss : %.2f\t\tTrain acc : %.2f\t\tValid acc : %.2f" % \
              (epoch + 1, epoch_train_loss, epoch_valid_loss, train_accuracy, valid_accuracy))
       
    torch.save(net.state_dict(), MODELPATH)
    
    inp_test, out_test = Variable(torch.from_numpy(x_test)).float().cpu(), Variable(torch.from_numpy(y_test)).long().cpu()
    test_sum = 0
    for i in range(0, TEST_SIZE, BATCH_SIZE):
        pred_test       = net(inp_test[i:i + BATCH_SIZE])
        indices_test    = pred_test.max(1)[1]
        test_sum        += (indices_test == out_test[i:i + BATCH_SIZE]).sum().data.cpu().numpy()[0]
    test_accuracy   = test_sum / float(TEST_SIZE)
    print("Test acc: %d " % test_accuracy)

    return

if __name__ == '__main__':
    main()


   #GET_GENRE

from collections import Counter

import warnings
warnings.filterwarnings("ignore")


def main(audio_path):  

    le = LabelEncoder().fit(GENRES)

    net         = genreNet()
    net.load_state_dict(torch.load(MODELPATH, map_location='cpu'))
    audio_path  = '/Users/mayil/Downloads/test_music.wav'
    print(audio_path)
    y, sr       = load(audio_path, mono=True, sr=22050)

    S           = melspectrogram(y, sr).T
    S           = S[:-1 * (S.shape[0] % 128)]
    num_chunk   = S.shape[0] / 128
    data_chunks = np.split(S, num_chunk)
    genres = list()
    for i, data in enumerate(data_chunks):
        data    = torch.FloatTensor(data).view(1, 1, 128, 128)
        preds   = net(data)
        pred_val, pred_index    = preds.max(1)
        pred_index              = pred_index.data.numpy()
        pred_val                = np.exp(pred_val.data.numpy()[0])
        pred_genre              = le.inverse_transform(pred_index).item()
        if pred_val >= 0.5:
            genres.append(pred_genre)
    s           = float(sum([v for k,v in dict(Counter(genres)).items()]))
    pos_genre   = sorted([(k, v/s*100 ) for k,v in dict(Counter(genres)).items()], key=lambda x:x[1], reverse=True)
    for genre, pos in pos_genre:
        print("%10s: \t%.2f\t%%" % (genre, pos))
    return

if __name__ == '__main__':
    audio_path  = '/Users/mayil/Downloads/test_music.wav'
    main(audio_path)
    