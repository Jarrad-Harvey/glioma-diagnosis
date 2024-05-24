Note: This is the readme which explains how to run the program. For the SVM discussion see the file `readme_svm.pdf`.


### How to run
1. Download the BraTS2020 data set from https://www.kaggle.com/datasets/awsaf49/brats2020-training-data/
2. Place the extracted folder at `./data`.
    - Use `sort_slices.py` to sort the dataset subfolders by volume.
3. Install *requirements.txt*.
    - `pip install -r requirements.txt`
    - Note: This project uses Python 3.9. Therefore, using an environment is recommended.
4. Run the GUI with `python GUI.py`.