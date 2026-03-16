import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np

class IEMOCAP4(Dataset):

    def __init__(self, path, train= True):
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoText,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
        ) = data

        if(train):
            self.currentVideos = self.trainVid
        else:
            self.currentVideos = self.testVid
        
        self.length = len(self.currentVideos)
        self.emotions = self.videoLabels
        
        sentiments_dict = {}

        for i in self.videoLabels:
            dialogue_sentiments = []
            for j in self.videoLabels[i]:
                if j in [1, 3]:
                    dialogue_sentiments.append(0)
                elif j in [2]:
                    dialogue_sentiments.append(1)
                elif j in [0]:
                    dialogue_sentiments.append(2)
            sentiments_dict[i] = dialogue_sentiments
        self.sentiments = sentiments_dict


    def __getitem__(self, index):
        
        videoID_list = self.currentVideos[index]

        return (
            torch.FloatTensor(np.array(self.videoText[videoID_list])),
            torch.FloatTensor(np.array(self.videoText[videoID_list])),
            torch.FloatTensor(np.array(self.videoText[videoID_list])),
            torch.FloatTensor(np.array(self.videoText[videoID_list])),
            torch.FloatTensor(np.array(self.videoVisual[videoID_list])),
            torch.FloatTensor(np.array(self.videoAudio[videoID_list])),
            torch.FloatTensor(
                [
                    [1, 0] if x == "M" else [0, 1]
                    for x in np.array(self.videoSpeakers[videoID_list])
                ]
            ),
            torch.FloatTensor([1] * len(np.array(self.emotions[videoID_list]))),
            torch.LongTensor(np.array(self.emotions[videoID_list])),
            torch.LongTensor(np.array(self.sentiments[videoID_list])),
            videoID_list,
        )

    def __len__(self):
        return self.length
    
    def collate_fn(self, data):

        unpadded_data = pd.DataFrame(data)
        result = []

        for i in unpadded_data:
            if i < 10:
                result.append(pad_sequence(unpadded_data[i]))
            else:
                result.append(unpadded_data[i].tolist())
        
        return result


class IEMOCAP(Dataset):

    def __init__(self, path, train= True):
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoText0,
            self.videoText1,
            self.videoText2,
            self.videoText3,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
        ) = data

        if(train):
            self.currentVideos = self.trainVid
        else:
            self.currentVideos = self.testVid
        
        self.length = len(self.currentVideos)
        self.emotions = self.videoLabels
        sentiments_dict = {}

        for i in self.videoLabels:
            dialogue_sentiments = []
            for j in self.videoLabels[i]:
                if j in [1, 3, 5]:
                    dialogue_sentiments.append(0)
                elif j in [2]:
                    dialogue_sentiments.append(1)
                elif j in [0, 4]:
                    dialogue_sentiments.append(2)
            sentiments_dict[i] = dialogue_sentiments
        self.sentiments = sentiments_dict

    def __getitem__(self, index):
        
        videoID_list = self.currentVideos[index]

        return (
            torch.FloatTensor(np.array(self.videoText0[videoID_list])),
            torch.FloatTensor(np.array(self.videoText1[videoID_list])),
            torch.FloatTensor(np.array(self.videoText2[videoID_list])),
            torch.FloatTensor(np.array(self.videoText3[videoID_list])),
            torch.FloatTensor(np.array(self.videoVisual[videoID_list])),
            torch.FloatTensor(np.array(self.videoAudio[videoID_list])),
            torch.FloatTensor(
                [
                    [1, 0] if x == "M" else [0, 1]
                    for x in np.array(self.videoSpeakers[videoID_list])
                ]
            ),
            torch.FloatTensor([1] * len(np.array(self.emotions[videoID_list]))),
            torch.LongTensor(np.array(self.emotions[videoID_list])),
            torch.LongTensor(np.array(self.sentiments[videoID_list])),
            videoID_list,
        )
    
    def __len__(self):
        return self.length
    
    def collate_fn(self, data):
        
        unpadded_data = pd.DataFrame(data)
        result = []

        for i in unpadded_data:
            if i < 10:
                result.append(pad_sequence(unpadded_data[i]))
            else:
                result.append(unpadded_data[i].tolist())
        
        return result
    
class MELD(Dataset):
    
    def __init__(self, path, train= True):
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoSentiments,
            self.videoText0,
            self.videoText1,
            self.videoText2,
            self.videoText3,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
            _,
        ) = data

        if(train):
            self.currentVideos = list(self.trainVid)
        else:
            self.currentVideos = list(self.testVid)
        
        self.length = len(self.currentVideos)
        self.emotions = self.videoLabels
        self.sentiments = self.videoSentiments

    def __getitem__(self, index):
        
        videoID_list = self.currentVideos[index]

        return (
            torch.FloatTensor(np.array(self.videoText0[videoID_list])),
            torch.FloatTensor(np.array(self.videoText1[videoID_list])),
            torch.FloatTensor(np.array(self.videoText2[videoID_list])),
            torch.FloatTensor(np.array(self.videoText3[videoID_list])),
            torch.FloatTensor(np.array(self.videoVisual[videoID_list])),
            torch.FloatTensor(np.array(self.videoAudio[videoID_list])),
            torch.FloatTensor(np.array(self.videoSpeakers[videoID_list])),
            torch.FloatTensor([1] * len(np.array(self.emotions[videoID_list]))),
            torch.LongTensor(np.array(self.emotions[videoID_list])),
            torch.LongTensor(np.array(self.sentiments[videoID_list])),
            videoID_list,
        )
    
    def __len__(self):
        return self.length
    

    def return_emotions(self):
        return_emotions = []
        for videoIds in self.currentVideos:
            return_emotions += self.videoLabels[videoIds]
        return return_emotions
    
    def collate_fn(self, data):
        
        unpadded_data = pd.DataFrame(data)
        result = []

        for i in unpadded_data:
            if i < 10:
                result.append(pad_sequence(unpadded_data[i]))
            else:
                result.append(unpadded_data[i].tolist())
        
        return result

    