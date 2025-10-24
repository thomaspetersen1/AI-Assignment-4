import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
# ......
# --- end of task --- #

# load an imbalanced data set 
# there are 50 positive class instances 
# there are 500 negative class instances 
data = np.loadtxt('diabetes_new.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]

# vary the percentage of data for training
num_train_per = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

acc_base_per = []
auc_base_per = []

acc_yours_per = []
auc_yours_per = []

# hyper parameter for the class_weights of the minority(positive) and find optimal
bonus_weights = [1, 2, 5, 10, 20]
auc_bonus_weights = []

for per in num_train_per: 

    # create training data and label
    num_train = int(n*per)
    sample_train = data[0:num_train,0:-1]
    label_train = data[0:num_train,-1]

    model = LogisticRegression()

    # --- Your Task --- #
    # Implement a baseline method that standardly trains 
    # the model using sample_train and label_train
    # ......
    # ......
    # ......
    model.fit(sample_train, label_train)

    # Predictions
    pred_test = model.predict(sample_test)
    pred_proba = model.predict_proba(sample_test)[:,1] # positive class
    
    # evaluate model testing accuracy and stores it in "acc_base"
    # ......
    acc_base = accuracy_score(label_test, pred_test)
    acc_base_per.append(acc_base)
    
    # evaluate model testing AUC score and stores it in "auc_base"
    # ......
    auc_base = roc_auc_score(label_test, pred_proba)
    auc_base_per.append(auc_base)
    # --- end of task --- #
    
    
    # --- Your Task --- #
    # Now, implement your method 
    # Aim to improve AUC score of baseline 
    # while maintaining accuracy as much as possible 
    # ......
    # ......
    # ......
    # evaluate model testing accuracy and stores it in "acc_yours"
    # ......

    # notes: class_weight helps handle imbalanced datasets and prevents bias during training
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(sample_train, label_train)

    # predictions
    pred_test = model.predict(sample_test)
    pred_proba = model.predict_proba(sample_test)[:,1]

    
    acc_yours = accuracy_score(label_test, pred_test)
    acc_yours_per.append(acc_yours)
    # evaluate model testing AUC score and stores it in "auc_yours"
    # ......
    auc_yours = roc_auc_score(label_test, pred_proba)
    auc_yours_per.append(auc_yours)
    # --- end of task --- #

for w in bonus_weights:
    # for the minority try the regression model with each weight to balance it
    model = LogisticRegression(class_weight={0:1, 1:w}) 
    model.fit(sample_train, label_train)
    pred_proba_bonus = model.predict_proba(sample_test)[:, 1]
    auc_score = roc_auc_score(label_test, pred_proba_bonus)
    auc_bonus_weights.append(auc_score)
    

plt.figure()    
plt.plot(num_train_per,acc_base_per, label='Base Accuracy')
plt.plot(num_train_per,acc_yours_per, label='Your Accuracy')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification Accuracy')
plt.legend()


plt.figure()
plt.plot(num_train_per,auc_base_per, label='Base AUC Score')
plt.plot(num_train_per,auc_yours_per, label='Your AUC Score')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification AUC Score')
plt.legend()

plt.figure()
plt.plot(bonus_weights, auc_bonus_weights, marker='o')
plt.xlabel('Minority Class Weight')
plt.ylabel('Classification AUC Score')
plt.title('Figure 6: Effect of Minority Class Weight on AUC')
plt.show()
    


