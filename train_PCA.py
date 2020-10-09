import numpy as np
import xgboost as xgb
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import argparse



from dataloader import *
from utility import *


parser = argparse.ArgumentParser()
parser.add_argument('-subject', type = str, default = 'M15')
args = parser.parse_args()

subject= args.subject

#################################################### data load ##########################################################

labels = np.arange(180)
labels_train = np.reshape(np.tile(labels,(42,1)),(7560))  
labels = np.reshape(np.tile(labels,(3,1)),(540))

class_fine_test =np.eye(180)
class_fine_test = np.reshape(np.tile(class_fine_test,(3,1)),(540,180))

data_train, data_test, glove_train, glove_test, data_fine, data_fine_test, glove_fine, glove_fine_test = dataloader_sentence_word_split_new_matching_all_subjects(subject=subject)

le_y = LabelEncoder()

class_fine_test_two = le_y.fit_transform(labels)
class_fine =le_y.transform(labels_train)

#################################################### feature selection ##########################################################


pca = PCA(n_components=5000)
pca.fit(data_fine)
data_train = pca.transform(data_fine)
data_test = pca.transform(data_fine_test)
print('PCA is done')
#################################################### XGBoost ##########################################################

classer = xgb.XGBClassifier(learning_rate =0.1,
                    n_estimators=5,
                    max_depth=3,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=1,
                    objective='multi:softprob',
                    seed=27)

classer.fit(data_fine, class_fine)

##################################################### Evaluation #######################################################

dtrainPredictions = classer.predict(data_fine_test)
proba = classer.predict_proba(data_fine_test)

pred_class = softmax(proba, axis=1)

accuracy, accuracy_five, accuracy_ten = top_5(pred_class,class_fine_test)
print('Accuracy:', accuracy, ' Accuracy_top5:', accuracy_five, 'Accuracy_top10:', accuracy_ten)