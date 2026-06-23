# DSTINet

**Experimental Settings:** All models were evaluated with batch size 32 for both training and inference on an NVIDIA GeForce RTX 4090 GPU. The training used learning rate 0.001 with the Ranger optimizer, and 500 epochs with early stopping patience of 100 epochs.

**Unified Framework:** All deep learning models were implemented and evaluated within a unified PyTorch framework to guarantee consistency. The dataset was partitioned into training, validation, and test sets using a strict 6:2:2 ratio, and identical data preprocessing and evaluation metrics were applied across the board. Furthermore, we ensured a fair comparison by maintaining a uniform configuration of 12 time steps for both input and output lengths across all models.

**Baseline Implementations:** For baseline methods, we adopted their official implementations from publicly available repositories when available. All baseline models were trained with their recommended hyperparameters from the original papers, and the complete parameter configurations are provided in *`params.json`* at <https://anonymous.4open.science/r/DSTINet-C16C> for reproducibility.
