"""
Replace the `PATH` variable with your path to the folder "small" from the "abo-images-small" tarball
"""
import pandas as pd, shutil, os

folder = 'images_houseware'

PATH = '/Users/grigoryturchenko/Downloads/abo-images-small 2/images/small'

metadata = pd.read_csv('../metadata/images_houseware.csv')

if not os.path.exists(folder):

    for category in metadata['category'].unique():

        label_folder = f'{folder}/{category}'
        os.makedirs(label_folder)

        for img_path in metadata[metadata['category'] == category]['path']:

            shutil.copy(f'{PATH}/{img_path}', label_folder)