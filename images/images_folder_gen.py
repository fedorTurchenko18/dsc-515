"""
Replace the `PATH` variable with your path to the folder "small" from the "abo-images-small" tarball
"""
import pandas as pd, shutil, os

folder = 'phone_cases_images'

if not os.path.exists(folder):
    os.makedirs(folder)

PATH = '/Users/grigoryturchenko/Downloads/abo-images-small/images/small'

metadata = pd.read_csv('../metadata/images_phone_cases.csv')['path']

for img_path in metadata:
    shutil.copy(f'{PATH}/{img_path}', folder)