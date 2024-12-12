#Import Files 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


#Loading the nba dataset for training 
nba_stats = pd.read_csv('nba_stats.csv')

if (nba_stats.isnull().sum().any()):
    print(f"There is a null or zero value present in the NBA Dataset:\n {nba_stats.isnull().sum()}")
else:
    print("There is no null or zero value in the NBA Dataset")

# Encoding only the Pos column 
lab = LabelEncoder()
y_set = lab.fit_transform(nba_stats['Pos'])

# Saving all the position labels 
position_labels = lab.classes_

# Only considering the Important features
x_set = nba_stats[['AST', 'STL', 'BLK', 'TRB', 'FGA', 'FG%', '3P', '3PA', 'PF', 'eFG%', 'FT%', 'DRB', '3P%', '2P', '2P%']]

# Spliting the training data into training and validation data 
x_train, x_val, y_train, y_val = train_test_split(x_set, y_set, test_size=0.2, random_state=0)

#Implementing the KNN 
knn = KNeighborsClassifier(n_neighbors=7, weights='uniform',algorithm = 'auto', p=2)
knn.fit(x_train, y_train)

y_train_prediction = knn.predict(x_train)
y_val_pre = knn.predict(x_val)

# Calculate accuracy 
training_accuracy = (accuracy_score(y_train, y_train_prediction) * 100)
val_accuracy = (accuracy_score(y_val, y_val_pre) * 100)


#Loading the Testing set for testing 
nba_dummy_testset = pd.read_csv('dummy_test.csv')
x_testing = nba_dummy_testset[['AST', 'STL', 'BLK', 'TRB', 'FGA', 'FG%', '3P', '3PA', 'PF', 'eFG%', 'FT%', 'DRB', '3P%', '2P', '2P%']]
y_testing = lab.transform(nba_dummy_testset["Pos"])

# Testing the Dummy Testset
y_test_predict = knn.predict(x_testing)
testingset_accuracy = (accuracy_score(y_testing, y_test_predict ) * 100)


position_accuracies = {}
for pos_index, pos_label in enumerate(position_labels):
    pos_indices = (y_testing == pos_index)  # Get indices for this position
    pos_accuracy = accuracy_score(y_testing[pos_indices], y_test_predict[pos_indices]) * 100
    position_accuracies[pos_label] = pos_accuracy


# Appling 10-fold stratified cross-validation.
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
cross_validation = cross_val_score(knn, x_set, y_set, cv=skf)

cv_mean = cross_validation.mean() 


print(f"\033[1mTraining accuracy using KNN:\033[0m {training_accuracy:.2f}%")
print(f"\033[1mValidation accuracy using KNN:\033[0m {val_accuracy:.2f}%")

print(f"Training Confusion Matrix on the nba_stats:\n {confusion_matrix(y_train,y_train_prediction)}")
print(f"Validation Confusion Matrix on the nba_stats: \n { confusion_matrix(y_val, y_val_pre)}")

print(f"\033[1mTesting accuracy using KNN on Dummyset(NBA):\033[0m {testingset_accuracy:.2f}%")

print(f"Dummy_dataset (Testing) Confusion Matrix:\n {confusion_matrix(y_testing ,y_test_predict )}")

print("Cross-validation scores for each fold:", cross_validation)
print(f"\033[1mMean Cross Validation Accuracy:\033[0m {(cv_mean * 100):.2f}%")



print("\nPosition-wise classification accuracy on the dummy test set:")
for position, accuracy in position_accuracies.items():
    print(f"{position}: {accuracy:.2f}%")

# References and articles used
# 1) https://www.geeksforgeeks.org/working-csv-files-python/
# 2) https://stackoverflow.com/questions/52400408/import-csv-file-into-python
# 3) https://www.statology.org/label-encoding-in-python/
# 4) https://www.geeksforgeeks.org/how-to-do-train-test-split-using-sklearn-in-python/
# 5) https://www.geeksforgeeks.org/k-nearest-neighbor-algorithm-in-python/
# 6) https://www.geeksforgeeks.org/confusion-matrix-machine-learning/
# 7) https://www.geeksforgeeks.org/stratified-k-fold-cross-validation/
# 8) https://stackoverflow.com/questions/62144055/how-can-i-find-the-isnull-value-in-csv-file
# 9) https://stackoverflow.com/questions/8924173/how-can-i-print-bold-text-in-python