## Data preparation

You need the following files:

 - `dissimilarities.zip`
 - `successfulAttacks.csv`
 
### Preparation

 - Unzip `dissimilarities.zip` inside this folder (`data/` folder):
 
    ```shell
    unzip dissimilarities.zip -d data/
    ```
 - Move `successfulAttacks.csv` inside the created `data/dissimilarities/` folder:
 
    ```shell
    mv successfulAttacks.csv data/dissimilarities/
    ```
 - Parse the data in `numpy` format:
    ```shell
    python data/parse_dissimilarities.py data/dissimilarities data/dissim.npz
    ```