# glioma-diagnosis

### How to run
1. Download the BraTS2020 data set from https://www.kaggle.com/datasets/awsaf49/brats2020-training-data/
2. Place the extracted folder at `./data`.
    - Use `sort_slices.py` to sort the dataset subfolders by volume.
3. Install *requirements.txt*.
    - In Conda: `conda create --name <env> --file requirements.txt`
    - Note: This project uses Python 3.9
4. Execute with `python glioma-diagnosis.py`.