import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
# ......
# --- end of task --- #

# load a data set for regression
# in array "data", each row represents a community 
# each column represents an attribute of community 
# last column is the continuous label of crime rate in the community
data = np.loadtxt('crimerate.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]

# --- Your Task --- #
# now, pick the percentage of data used for training 
# remember we should be able to observe overfitting with this pick 
# note: maximum percentage is 0.75 
per = .6
num_train = int(n*per)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
# --- end of task --- #


# --- Your Task --- #
# We will use a regression model called Ridge. 
# This model has a hyper-parameter alpha. Larger alpha means simpler model. 
# Pick 5 candidate values for alpha (in ascending order)
# Remember we should aim to observe both overfitting and underfitting from these values 
# Suggestion: the first value should be very small and the last should be large 
alpha_vec = [1e-4, 1e-2, 1e-1, 1, 10]
# --- end of task --- #

er_train_alpha = []
er_test_alpha = []

k = 5
fold_size = num_train // k
er_valid_alpha = []

for alpha in alpha_vec: 
    valid_alphas = []
    # pick ridge model, set its hyperparameter 
    model = Ridge(alpha = alpha)
    
    # --- Your Task --- #
    # now implement k-fold cross validation 
    # on the training set (which means splitting 
    # training set into k-folds) to get the 
    # validation error for each candidate alpha value 
    # store it in "er_valid"
    # ......
    # ......
    # ......
    # ......
    for i in range(k):

        # get index for both start and end of each fold.
        start = i * fold_size # current fold * the size of each fold
        # either the beginning of the next fold or the end of the training set
        end = min((i + 1) * fold_size, num_train) 

        # treat current fold as the validation set, all others are the training set
        val_x = sample_train[start:end]
        val_y = label_train[start:end]

        # training set, everything before and after the current fold
        train_x = np.concatenate((sample_train[:start], sample_train[end:]), axis=0)
        train_y = np.concatenate((label_train[:start], label_train[end:]), axis=0)

        #train model on current training set
        model.fit(train_x, train_y)

        # validation error for current fold
        pred_val = model.predict(val_x)
        valid_alphas.append(mean_squared_error(val_y, pred_val))

    er_valid = np.mean(valid_alphas) # mean of each fold's error
    er_valid_alpha.append(er_valid)
    # --- end of task --- #


# Now you should have obtained a validation error for each alpha value 
# In the homework, you just need to report these values

# The following practice is only for your own learning purpose.
# Compare the candidate values and pick the alpha that gives the smallest error 
# set it to "alpha_opt"
alpha_opt = alpha_vec[np.argmin(er_valid_alpha)] # get the valid alpha that had the least error

# now retrain your model on the entire training set using alpha_opt 
# then evaluate your model on the testing set 
model = Ridge(alpha = alpha_opt)
# ......
# ......
# ......
model.fit(sample_train, label_train)
pred_train = model.predict(sample_train)
pred_test = model.predict(sample_test)

er_train = mean_squared_error(label_train, pred_train)
er_test = mean_squared_error(label_test, pred_test)

print("Alpha\t\tValidation Error")
print("-" * 30)
for alpha, error in zip(alpha_vec, er_valid_alpha):
    print(f"{alpha:<10}\t{error:.4f}")

print(f"Optimal Alpha: {alpha_opt}")
print(f"Predicted Training MSE: {er_train}")
print(f"Predicted Testing MSE: {er_test}")


