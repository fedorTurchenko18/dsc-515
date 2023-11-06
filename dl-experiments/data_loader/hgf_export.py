import cv2, numpy as np, pandas as pd
from typing import Tuple, Literal
from huggingface_hub import login, logout
from bertopic import BERTopic
from datasets import load_dataset


class HGFresource:

    def __init__(self, token):
        # keep private
        self.__token = token

    def load_data(self, repo: str, sets: Literal['train', 'test', 'all']) -> Tuple[np.ndarray]:
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

        `Tuple[np.ndarray]`
            If `sets='train'` or `sets='test'`, returns corresponding images (converted to array) and labels, i.e. `images, labels` \n
            Otherwise, returns all train and test images (converted to array) and labels, i.e. `train_images, train_labels, test_images, test_labels`
        '''

        dataset = load_dataset(repo, token=self.__token)
        
        if sets == 'all':
            train = dataset['train'].to_pandas()
            # convert images from bytes to numpy array
            train_images = train['image'].apply(
                          # by default RGB images produce 3d arrays
                          # but since we want final array of all images to be 4d
                          # we add an empty dimension along which images will be concatenated
                lambda x: np.expand_dims(
                    cv2.imdecode(np.frombuffer(x['bytes'], np.uint8), -1),
                    0
                )
            )
            # remove images with invalid dimensions
            train_images = train_images.apply(lambda x: x if len(x.shape) == 4 else np.nan)
            train_images.dropna(inplace=True)
            # reset index for proper concatenation
            train_images.reset_index(drop=True, inplace=True)
            # concatenate with float data type so that further preprocessing can be implemented
            train_images = np.concatenate(train_images, dtype=np.float16)
            train_labels = train['label'].values

            test = dataset['test'].to_pandas()
            test_images = test['image'].apply(
                lambda x: np.expand_dims(
                    cv2.imdecode(np.frombuffer(x['bytes'], np.uint8), -1),
                    0
                )
            )
            test_images = test_images.apply(lambda x: x if len(x.shape) == 4 else np.nan)
            test_images.dropna(inplace=True)
            test_images.reset_index(drop=True, inplace=True)
            test_images = np.concatenate(test_images, dtype=np.float16)
            test_labels = test['label'].values

            return train_images, train_labels, test_images, test_labels
        else:
            df = dataset[sets].to_pandas()
            images = df['image'].apply(
                lambda x: np.expand_dims(
                    cv2.imdecode(np.frombuffer(x['bytes'], np.uint8), -1),
                    0
                )
            )
            images = images.apply(lambda x: x if len(x.shape) == 4 else np.nan)
            images.dropna(inplace=True)
            images.reset_index(drop=True, inplace=True)
            images = np.concatenate(images, dtype=np.float16)
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