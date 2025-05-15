_Fine-Tuning and Comparing Pre-Trained CNN Architecture for Pet Breed Classification_
# MATH 392 Final Project
## Overview:
For this project, we will be fine-tuning and comparing at least two pre-trained CNN models to classify cat and dog images into 37 breeds. For our CNN models we will be using  **MoblieNetV3** and **ResNet18** to do the classification on the dataset Oxford-IIT Pet. 
## Final Results Summary

Each model below was fine tuned using GPU in Google Colab. Results are shown from hyperparameter optimization, final training, and evaluation phases.

| SETID | Model                    | LR    | Epochs | Accuracy | F1 Score | Precision | Recall | Loss | Notes |
|-------|--------------------------|-------|--------|----------|----------|-----------|--------|------|-------|
| 001   | ResNet18 (mid_aug)       |       |        |          |          |           |        |      |       |
| 002   | ResNet18 (noaug)         |       |        |          |          |           |        |      |       |
| 003   | MobileNetV2 (mid_aug)    |       |        |          |          |           |        |      |       |
| 004   | MobileNetV2 (noaug)      |       |        |          |          |           |        |      |       |
| 005   | ResNet18 (head_aug)      |       |        |          |          |           |        |      |       |
| 006   | ResNet18 (head_noaug)    |       |        |          |          |           |        |      |       |
| 007   | MobileNetV2 (head_aug)   |       |        |          |          |           |        |      |       |
| 008   | MobileNetV2 (head_noaug) |       |        |          |          |           |        |      |       |

---

## How to Run This Project

Everything you need is included in the repo and ready to run in Google Colab with GPU enabled.

### 1. Clone the Repository

git clone
cd MATH392_Final_Project

### 2. Run Hyperparameter Optimization
python scripts/run_hpo.py \
  --config configs/<SETIDs> \
  --epochs: 20
  --batch_size 64

### 3. Train the Final Model
Once the best hyperparameters are known, train your model with:
python scripts/run_train.py \
  --config configs/<SETIDs> \
  --batch_size 64
  --epochs 20 
  --lr 1e-4

### 4. Evaluate the Final Model
To evaluate the trained model, pass in the config and final checkpoint file:
python scripts/run_evaluation.py \
  --config configs/<SETIDs> \
  --checkpoint results/final_training/<SETIDs>/model_best.pth

### 5. Or Use the Jupyter Notebook
You can also run the entire pipeline from within a notebook:
notebooks/FinalNotebook.ipynb


