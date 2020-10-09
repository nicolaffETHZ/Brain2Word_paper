#!/bin/bash


#Comment out all the models and only uncomment the model you want to run. Set the parameters the way you want to run the model.

# python train_PCA.py -subject='M15' -class_model = 1
python train_ours.py -subject='M15' -class_model = 1
# python train_big.py -subject='M15' -class_model = 1
# python train_small.py -subject='M15' -class_model = 1


#Run loop over all the subjects. Change the run file accoring to the model you want to run.

# for variable in P01  M02  M03  M04  M05  M06  M07  M08  M09  M15  M10  M13  M14  M16  M17
# do
#     python train_ours.py -subject=variable -class_model = 1
# done

