# Calibration

repository for the paper "Removal of Batch Effects using Distribution-Matching Residual Networks" by Uri Shaham, Kelly P. Stanton, Jun Zhao, Huamin Li, Khadir Raddassi, Ruth Montgomery, and Yuval Kluger.

The script Train_MMD_ResNet.py is the main demo script, and can be generally used for calibration experiments. It was used to train all MMD ResNets used for the CyTOF experiments reported in our manuscript.
It loads two CyTOF datasets, corresponding to measurements of blood of the same person on the same machine in two different days, and denoises them. The script then trains a MMD-ResNet using one of the datasets as source and the other as target, to remove the batch effects. 

The CyTOF datasets used to produce the results in the manuscript are saved in Data.
The labels for the CyTOF datasets (person_Day_) were used only to separate the CD8 population during evaluation. Training of all models was unsupervised.
The RNA data set Data2_standardized_37PCs.csv contains the projection of the cleaned and filtered data onto the subspace of the first 37 principal components. To obtain the raw data please contact Jun Zhao at jun.zhao@yale.edu.  

All the models used to produce the results in the manuscript are saved in savedModels.


Any questions should be referred to Uri Shaham, uri.shaham@yale.edu.

# Usage
To use the command line interface version of our model, please run:
```
cd path_to_BatchEffectRemoval/src; python cmdline_MMD_ResNet.py --source_path=[SOURCE_PATH] --target_path=[TARGET_PATH]
```
where [SOURCE_PATH] and [TARGET_PATH] are the paths to the source dataset and target dataset respectively. The model will then calibrate the source dataset to match the target dataset. Alternatively, running the file without any arguments will load and demo the model on a default dataset.

cmdline_MMD_ResNet has many more arguments. These include the number of epochs to train, the width and depth of the ResNet architecture, and the learning rate schedule. To get a brief overview of each hyperparameter, please run with the '-h' flag:
```
python cmdline_MMD_ResNet.py -h
```
