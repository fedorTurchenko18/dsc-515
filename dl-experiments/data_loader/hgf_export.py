import cv2, numpy as np, pandas as pd, random, tensorflow as tf, pickle, json
from typing import Tuple, Literal, List, Optional, Union
from huggingface_hub import login, logout, hf_hub_download, HfApi, CommitOperationAdd
from bertopic import BERTopic
from top2vec import Top2Vec
from datasets import load_dataset
from collections import Counter


class HGFresource:

    def __init__(self, token: str):
        '''
        token: str
            HuggingFace access token generated at https://huggingface.co/settings/tokens
        '''
        # keep private
        self.__token = token


    def load_data_tfds(
            self,
            repo: str,
            n_classes: int,
            batch_size: int,
            perform_split: Optional[bool] = False,
            train_size: Optional[float] = None
    ) -> Tuple:
        '''
        Load datasets from Hugging Face organization repository as a Tensorflow MapDataset object

        Parameters
        ----------

        repo : str
            Path to the data repo on Hugging Face

        n_classes: Optional[int]
            Number of classes of target variable \n
            Different versions of dataset have different one, so should be specified

        batch_size : int
            Desired batch size for the model training

        perform_split : bool = False
            Whether to perform train-validation split of the training set. No split by default

        train_size : float = None
            If split is performed, the share of the training subset
        Returns
        -------

        Since the object returned by this function is somewhat unfamiliar (both images and labels combined into a single structure),
        below is the quickstart code example (also available in the load_as_tfds.ipynb):

        ```
        import os, numpy as np, tensorflow as tf
        from data_loader.hgf_export import HGFresource
        from dotenv import load_dotenv
        load_dotenv()

        HGF_TOKEN = os.environ['HUGGINGFACE_TOKEN']
        HGF_DATA_REPO = os.environ['HUGGINGFACE_DATASET_REPO']
        HGF_TOPIC_MODEL_REPO = os.environ['HUGGINGFACE_TOPIC_MODEL_REPO']

        hgf = HGFresource(token=HGF_TOKEN)

        # Default data load
        train_data, test_data = hgf.load_data_tfds(repo=HGF_DATA_REPO, batch_size=32)

        # Data load with additional split of the train data
        train_data_split, val_data_split, test_data_split = hgf.load_data_tfds(repo=HGF_DATA_REPO, batch_size=32, perform_split=True, train_size=0.7)

        OPTIMIZER = 'adam'
        LOSS = 'categorical_crossentropy'
        METRICS = [tf.keras.metrics.F1Score('weighted')]

        EPOCHS = 20

        INPUT_SHAPE = (256, 219, 3)
        N_CLASSES = 29

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(N_CLASSES, activation='softmax')
            ]
        )

        model.compile(
            OPTIMIZER,
            LOSS,
            METRICS
        )

        history = model.fit(
            train_data,
            epochs=EPOCHS,
            validation_data=test_data
        )
        ```
        '''
        dataset = load_dataset(repo, token=self.__token)
        train_data = dataset['train']
        test_data = dataset['test']

        if perform_split == False:
            train_data = self.__process_dataset(train_data, n_classes, batch_size, perform_split, train_size)
            test_data = self.__process_dataset(test_data, n_classes, batch_size, perform_split, train_size)
            return train_data, test_data

        else:
            train_data, val_data = self.__process_dataset(train_data, batch_size, True, train_size)
            test_data = self.__process_dataset(test_data, batch_size, False, None)
            return train_data, val_data, test_data


    def __process_dataset(self, dataset, n_classes, batch_size, perform_split, train_size):
        def preprocess_images(examples):
            examples['image'] = [tf.cast(image.convert('RGB'), tf.float32) / 255.0 for image in examples['image']]
            return examples

        def preprocess_labels(example):
            zeros = np.zeros(n_classes)
            np.put(zeros, example, 1)
            zeros = tf.convert_to_tensor(zeros)
            example = {'label': zeros}
            return example

        if perform_split == False:
            dataset = dataset.map(preprocess_labels, input_columns=['label'])
            dataset = dataset.with_transform(preprocess_images, ['image'], True)

            dataset = dataset.to_tf_dataset(
                columns='image',
                label_cols='label',
                batch_size=batch_size,
                shuffle=True,
                prefetch=False
            )
            return dataset

        else:
            dataset = dataset.train_test_split(train_size=train_size, stratify_by_column='label', seed=515)

            train = dataset['train']
            test = dataset['test']

            train = train.map(preprocess_labels, input_columns=['label'])
            train = train.with_transform(preprocess_images, ['image'], True)

            train = train.to_tf_dataset(
                columns='image',
                label_cols='label',
                batch_size=batch_size,
                shuffle=True,
                prefetch=False
            )

            test = test.map(preprocess_labels, input_columns=['label'])
            test = test.with_transform(preprocess_images, ['image'], True)

            test = test.to_tf_dataset(
                columns='image',
                label_cols='label',
                batch_size=batch_size,
                shuffle=True,
                prefetch=False
            )
            return train, test


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
            # assure reproducibility
            random.seed(42)
            # sample indices for the given class
            sampled_data_indices.extend(
                random.sample(class_data, samples_to_take)
            )
        return sampled_data_indices



    def load_model(self, repo: str, filename: Optional[Union[str, dict]] = None, model_type: Literal['seq', 'func'] = None) -> Union[BERTopic, Top2Vec, tf.keras.Sequential]:
        '''
        Load model from Hugging Face organization repository

        Parameters
        ----------
        repo : str
            Path to the model repo on Hugging Face

        filename : Optional[Union[str, dict]]
            Name of the file inside the model repo on Hugging Face \n
            It is required for loading files which cannot be exported via
            `huggingface_hub` methods \n
            If `dict` (required for model building workflow):
            ```
            {
                'model_weights': 'model_weights_filename.h5',
                'model_config': 'model_config_filename.json'
            }
            ```

        Returns
        -------

        Model of interest
        '''
        # custom method is implemented for each type of model loading
        # due to heterogeneity of their implementations
        # in terms of integration with huggingface
        if 'top2vec' not in repo and 'topic' in repo:
            # BERTopic model
            model = self.__load_bertopic_model(repo)
        elif 'top2vec' in repo:
            # Top2Vec model
            model = self.__load_top2vec_model(repo, filename)
        else:
            # deep learning model
            model = self.__load_dl_model(repo, filename, model_type)

        return model
    

    def load_file(self, repo: str, filename: str):
        '''
        Download any single file from specific repo

        Parameters
        ----------

        repo : str
            Path to the file repo on Hugging Face

        filename : str
            Name of the file of interest inside the repo

        Returns
        -------

        File of interest
        '''
        file = hf_hub_download(repo_id=repo, filename=filename, token=self.__token)
        return file
    

    def __load_bertopic_model(self, repo: str) -> BERTopic:
        '''
        Load BERTopic model from Hugging Face organization repository

        Parameters
        ----------
        repo : str
            Path to the BERTopic model repo on Hugging Face

        Returns
        -------

        Trained BERTopic model
        '''
        login(token=self.__token)
        model = BERTopic.load(repo)
        logout()

        return model


    def __load_top2vec_model(self, repo: str, filename: str) -> Top2Vec:
        '''
        Load top2vec model from Hugging Face organization repository

        Parameters
        ----------
        repo : str
            Path to the top2vec model repo on Hugging Face

        filename : str
            Name of the file with serialized top2vec model inside the repo on Hugging Face

        Returns
        -------

        Trained Top2Vec model
        '''
        model_file = self.load_file(repo=repo, filename=filename)
        with open(model_file, 'rb') as file:
            model = pickle.load(file)

        return model
    

    def __load_dl_model(self, repo: str, filename: dict, model_type: Literal['seq', 'func']) -> tf.keras.Sequential:
        '''
        Load deep learning model from Hugging Face organization repository

        Parameters
        ----------
        repo : str
            Path to the deep learning model repo on Hugging Face

        filename : dict
            Dictionary of a form:
            ```
            {
                'model_weights': 'model_weights_filename.h5',
                'model_config': 'model_config_filename.json'
            }
            ```

        model_type : Literal['seq', 'func']
            Either Sequential or Functional API based model to load
        
        Returns
        -------

        Deep learning model
        '''
        # download weights
        weights_file = self.load_file(repo=repo, filename=filename['model_weights'])
        # download config
        model_config_file = self.load_file(repo=repo, filename=filename['model_config'])
        # parse config
        with open(model_config_file, 'r') as json_file:
            model_config = json_file.read()
        model_config = json.loads(model_config)

        if model_type == 'seq':
            model = self.sequential_model_from_config(model_config, weights_file)
            return model, weights_file
        else:
            inputs, outputs = self.functional_model_from_config(model_config)
            return inputs, outputs, weights_file
    
    
    def functional_model_from_config(self, config):
        layers = config['layers']
        deserialized_layers = [tf.keras.layers.deserialize(layer) for layer in layers]

        input_layer = deserialized_layers[0]
        outputs = input_layer.output
        for dszd_layer in deserialized_layers:
            outputs = dszd_layer(outputs)
        inputs = input_layer.input

        return inputs, outputs
    

    def sequential_model_from_config(self, config):
        # re-create model
        model = tf.keras.Sequential.from_config(config)
        return model
    

    def commit_to_hub(self, repo: str, path_on_local: List[str], path_in_repo: List[str], commit_message: str) -> None:
        '''
        Commit changes to repository

        Parameters
        ----------

        repo : str
            Path of the repo on Hugging Face which to commit to

        path_on_local : List[str]
            List of local paths of files which were added/modified locally

        path_in_repo : List[str]
            List of remote paths of files which should be created/modified in repo

        commit_message : str
            Message which describes the files creations/modifications
        '''
        login(token=self.__token)

        api = HfApi()
        # no other operation than add/modify is needed
        operations = [
            CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local_path) for repo_path, local_path in zip(path_in_repo, path_on_local)
        ]
        api.create_commit(
            repo_id=repo,
            operations=operations,
            commit_message=commit_message,
        )

        logout()