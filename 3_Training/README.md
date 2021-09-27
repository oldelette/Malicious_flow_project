# You need to follow the below steps
## 1. Download the 1_Pcap from this link
https://drive.google.com/file/d/1orBl6xpb7swh1HJj-8zOg62hVlBVGqSe/view?usp=sharing

## 2. Edit the config.py
If you use the custom pcap dataset, you need the edit the following contents in config.py
1. class_num
2. class_dir
3. labels
4. dic 
5. zeroshot_class_dir 
6. zeroshot_labels 
7. zeroshot_dic 

## 3. run extract_feature.sh
After you run extract_feature.sh, you will get 4 files in 2_Csv/. There are
1. "train_joy.csv"
2. "test_joy.csv"
3. "train_handcraft.csv"
4. "test_handcraft.csv"

## 4. run training.sh
After you run training.sh, you will get weight files in 4_Weight/ and confusion matrix in 5_Result/

There are total 7 different machine learning algortihms (lightgbm, catboost, xgboost, svm, knn, decision tree, random forest) and three different feature sets (joy only, handcraft only, joy+handcraft)

## 5. run analyse.sh
After you run analyse.sh, you will get important and unimportant features distribution histogram in 6_Analysis/

There are important features in joy, unimportant features in joy, important features in handcraft, unimportant features in handcraft

## 6. run zeroshot.sh
After you run zeroshot.sh, you will get weight files in 4_Weight/ and confusion matrix in 5_Result/
