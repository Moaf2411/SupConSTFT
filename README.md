# SupConSTFT
A supervised contrastive learning model with three lanes to process STFT images extracted from EEG signal.

    1. a high frequency feature extraction lane
    
    2. a low frequency feature extraction lane
    
    3. a temporal feature extraction lane


In addition, a training module is implemented in this code, where checkpoints are saved on disc for future use, different metrics are also saved to a local database (SQLite) for model evaluation.
   

Below a TSNE plot from extracted representations is depicted, where the input data was the C3 EEG channel and the labels are sleep stages.

![image](https://github.com/user-attachments/assets/5ea55b4c-bee4-4a09-aec0-d84728f3a72d)
