import cv2, numpy as np, pandas as pd
from typing import Tuple, Literal
from huggingface_hub import login, logout
from bertopic import BERTopic
from datasets import load_dataset


class HGFresource:

    def __init__(self, token):
        # keep private
        self.__token = token

    def load_data(self, repo: str, sets: Literal['train', 'test', 'all']) -> Tuple[pd.Series]:
        '''
        Load dataset(s) from Hugging Face organization repository

        Parameters
        ----------

        repo : str
            Path to the data repo on Hugging Face

        sets : Literal['train', 'test', 'all']
            Data sets to export \n
            Possible options: `['train', 'test', 'all']`

        Returns
        -------

        `Tuple[pd.Series]`
            If `sets='train'` or `sets='test'`, returns corresponding images (converted to array) and labels, i.e. `images, labels` \n
            Otherwise, returns all train and test images (converted to array) and labels, i.e. `train_images, train_labels, test_images, test_labels`
        '''

        dataset = load_dataset(repo, token=self.__token)
        
        if sets == 'all':
            train = dataset['train'].to_pandas()
            train_images = train['image'].apply(lambda x: cv2.imdecode(np.frombuffer(x['bytes'], np.uint8), -1))
            train_labels = train['label'].values

            test = dataset['test'].to_pandas()
            test_images = test['image'].apply(lambda x: cv2.imdecode(np.frombuffer(x['bytes'], np.uint8), -1))
            test_labels = test['label'].values

            return train_images, train_labels, test_images, test_labels
        else:
            df = dataset[sets].to_pandas()
            images = df['image'].apply(lambda x: cv2.imdecode(np.frombuffer(x['bytes'], np.uint8), -1))
            labels = df['label'].values
            return images, labels


    def load_model(self, repo: str) -> BERTopic:
        '''
        Load model from Hugging Face organization repository

        Parameters
        ----------
        repo : str
            Path to the model repo on Hugging Face

        Returns
        -------

        Model of interest
        '''
        # for now only BERTopic model is in use
        # DL models will be saved to Hugging Face later
        # implementation of downloading these will be executed here
        if 'topic' in repo:
            login(token=self.__token)
            model = BERTopic.load(repo)
            logout()
        
        return model