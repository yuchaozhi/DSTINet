# DSTINet

**Experimental Settings:** All models were evaluated with batch size 32 for both training and inference on an NVIDIA GeForce RTX 4090 GPU. The training used learning rate 0.001 with the Ranger optimizer, and 500 epochs with early stopping patience of 100 epochs.

**Unified Framework:** All deep learning models were implemented and evaluated within a unified PyTorch framework to guarantee consistency. The dataset was partitioned into training, validation, and test sets using a strict 6:2:2 ratio, and identical data preprocessing and evaluation metrics were applied across the board. Furthermore, we ensured a fair comparison by maintaining a uniform configuration of 12 time steps for both input and output lengths across all models.

**Baseline Implementations:** For baseline methods, we adopted their official implementations from publicly available repositories when available. All baseline models were trained with their recommended hyperparameters from the original papers, and the complete parameter configurations are provided in *`params.json`* at <https://anonymous.4open.science/r/DSTINet-C16C> for reproducibility.

**Efficiency Metrics:** Table 5 reports parameter count, GFLOPs, GPU memory usage, training time per epoch, and inference latency on the validation set. As shown, DSTINet achieves the lowest parameter count (281K), GFLOPs (0.084G), GPU memory (86 MB), and inference latency (0.40s) among all compared models, while maintaining competitive predictive accuracy. This demonstrates that DSTINet achieves superior efficiency suitable for deployment in resource-constrained environments.
