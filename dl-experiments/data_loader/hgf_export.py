import cv2, numpy as np, pandas as pd, random
from typing import Tuple, Literal, List
from huggingface_hub import login, logout
from bertopic import BERTopic
from datasets import load_dataset
from collections import Counter


class HGFresource:

    def __init__(self, token):
        # keep private
        self.__token = token

    def load_data(self, repo: str, sets: Literal['train', 'test', 'all'], sample_fraction: float) -> Tuple[np.ndarray]:
        '''
        Load dataset(s) from Hugging Face organization repository

        Parameters
        ----------

        repo : str
            Path to the data repo on Hugging Face

        sets : Literal['train', 'test', 'all']
            Data sets to export \n
            Possible options: `['train', 'test', 'all']`

        sample_fraction : float
            Share of the dataset to return

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
            nan_indices = train_images[train_images.isna()].index
            train_images.dropna(inplace=True)
            ## reduce the size of data
            # extract labels as series
            train_labels = train['label']
            # remove NaN records from train labels as well
            train_labels = train_labels[~train_labels.index.isin(nan_indices)]
            # perform sampling
            TRAIN_INDICES_REMAINED = self.__sample_from_df(labels=train_labels, sample_fraction=sample_fraction)
            train_images = train_images[train_images.index.isin(TRAIN_INDICES_REMAINED)]
            # reset index for proper concatenation
            train_images.reset_index(drop=True, inplace=True)
            # concatenate with float data type so that further preprocessing can be implemented
            train_images = np.concatenate(train_images, dtype=np.float16)
            train_labels = train_labels[train_labels.index.isin(TRAIN_INDICES_REMAINED)].values

            test = dataset['test'].to_pandas()
            test_images = test['image'].apply(
                lambda x: np.expand_dims(
                    cv2.imdecode(np.frombuffer(x['bytes'], np.uint8), -1),
                    0
                )
            )
            test_images = test_images.apply(lambda x: x if len(x.shape) == 4 else np.nan)
            nan_indices = test_images[test_images.isna()].index
            test_images.dropna(inplace=True)
            test_labels = test['label']
            test_labels = test_labels[~test_labels.index.isin(nan_indices)]
            TEST_INDICES_REMAINED = self.__sample_from_df(labels=test_labels, sample_fraction=sample_fraction)
            test_images = test_images[test_images.index.isin(TEST_INDICES_REMAINED)]
            test_images.reset_index(drop=True, inplace=True)
            test_images = np.concatenate(test_images, dtype=np.float16)
            test_labels = test_labels[test_labels.index.isin(TEST_INDICES_REMAINED)].values

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
            nan_indices = test_images[test_images.isna()].index
            images.dropna(inplace=True)
            labels = df['label']
            labels = labels[~labels.index.isin(nan_indices)]
            INDICES_REMAINED = self.__sample_from_df(labels=test_labels, sample_fraction=sample_fraction)
            images = images.loc[INDICES_REMAINED]
            images.reset_index(drop=True, inplace=True)
            images = np.concatenate(images, dtype=np.float16)
            labels = labels.loc[INDICES_REMAINED].values
            return images, labels
        
    
    def __sample_from_df(self, labels: pd.Series, sample_fraction: float = 0.5) -> List[int]:
        '''
        Reduce the size of original dataframe through random stratified sampling

        Parameters
        ----------

        labels : pd.Series
            Either train or test labels
        
        sample_fraction : float
            Share of the dataset to return

        Returns
        -------

        sampled_data_indices : List[int]
            Sampled indices of `labels` Series \n
            They can be used to filter the train/test images/labels through `.loc` method
        '''
        # compute the class distribution in the original dataset
        target_freqs = Counter(labels)
        N_ROWS = labels.shape[0]
        # adjust to your desired dataset size
        DESIRED_SAMPLES = N_ROWS * sample_fraction

        # calculate the number of samples to take for each class
        sampling_ratios = {
            label: int(DESIRED_SAMPLES * count / N_ROWS)
            for label, count in target_freqs.items()
        }

        sampled_data_indices = []
        for label, count in target_freqs.items():
            class_data = labels[labels==label].index.tolist()
            samples_to_take = min(sampling_ratios[label], count)

            # sample indices for the given class
            sampled_data_indices.extend(
                random.sample(class_data, samples_to_take)
            )
        return sampled_data_indices



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