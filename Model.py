
import pandas as pd
import numpy as np


def PE(y,ypred):
    # Calculates the percentage error(PE)
    return (ypred-y)/y

def MdPE(y,ypred):
    #Calculates the MdPE(Median of Actual Error) - defined as the median of the percentage error
    return np.median(PE(y,ypred))

def MdAPE(y,ypred):
    # Calculates the MdAPE(Median Absolute Percentages Error)
    return np.median(abs(PE(y,ypred)))

def PPE(y,ypred,p): 
    # Calculates the proportion of models that have an error within a p% tolerance
    # of the true estimate
    a_pes = abs(PE(y,ypred))
    return np.sum(a_pes <= p)/len(y)

def evaluate(y,ypred):
    # This Function checks if each of the metrics above pass the criteria
    # specified in the project description
    
    mdpe = MdPE(y,ypred)
    mdape = MdAPE(y,ypred)
    ppe10 = PPE(y,ypred,0.1)
    ppe20 = PPE(y,ypred,0.2)

    print('MdPE:  {0:.3f}'.format(mdpe),', Pass =',bool(abs(mdpe) <= 0.03 ))
    print('MdAPE: {0:.3f}'.format(mdape),', Pass =',bool(mdape < 0.1 ))
    print('PPE10: {0:.3f}'.format(ppe10),', Pass =',bool(ppe10 > .7 ))
    print('PPE20: {0:.3f}'.format(ppe20),', Pass =',bool(ppe20 > .85 ))

# The HousingPriceIndex.csv data is taken from data.cso.ie 
#It contains quarterly information on the housing price index in ireland from 2010 - present

ind = pd.read_csv('HousingPriceIndex.csv')
new = list(np.where(ind['Dwelling Status'] == 'New')[0])
ex = list(np.where(ind['Dwelling Status'] == 'Existing')[0])

hi = pd.DataFrame(index = ind['Quarter'][new].values)
hi['Existing'] = ind['VALUE'][ex].values
hi['New'] = ind['VALUE'][new].values


# Reading in the data
house = pd.read_excel('jul20-dataset.xlsx')

#Making the new_home_indicator variable categorical - simplifies integration of the house price index
house['new_home_ind'] = np.where(house['new_home_ind'] == 1,'New','Existing') # setting this to categorical

# Creating a Quarter Column to be used for integrating the house price index 
house['Quarter'] = pd.PeriodIndex(house['sold_date'], freq='Q').astype(str)

# This code crates a column 'sold_price_2015' which scales the price sold_price to 2015 levels.
price = []
for q ,t in zip(house['Quarter'].values,house['new_home_ind'].values):
    price.append(hi.loc[q,t]/100)
    
house['Price_Index'] = price
house['sold_price_2015'] = house['sold_price']/house['Price_Index']

# List of relevant categorical variables to be used
cat_var_names = ['eircode_area_code','new_home_ind','building_property_subtype_code']

#one hot encoding the categorical variables, necessary for the regression model
cat_vars = pd.get_dummies(house[cat_var_names])

#Column names of the relevant numerical features
cols = ['building_shape_area_value','age_when_sold','sqm_value','yoc_value','beds_value','baths_value']
num_feats = house[cols]

# Concatenating the one hot encoded categorical variables dataframe with
# the numerical feature data frame

X = pd.concat((
    num_feats,
    cat_vars), axis = 1)

y = house['sold_price_2015'].values # What will be used in the modelling process
y_true = house['sold_price'].values # The actual prices of the house being sold

# The Knn imputer fills in missing values 'Nans' with the values of a data points 5 nearest neighbors
from sklearn.impute import KNNImputer
imputer = KNNImputer()

X = pd.DataFrame(imputer.fit_transform(X),columns = list(X.columns))

# Importing the relevant modules
import xgboost
from xgboost import plot_importance
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

#This scorerer is used to find the best score for the hyperparamter tuning further on in the analysis.

def my_scorer(y,ypred):
    mdpe = MdPE(y,ypred)
    mdape = MdAPE(y,ypred)
    ppe10 = PPE(y,ypred,0.1)
    ppe20 = PPE(y,ypred,0.2)
    
    score = int(abs(mdpe) <= 0.03 ) + int(mdape < 0.1 ) + int(ppe10 > .7 ) + int(ppe20 > .85)
    score += (1-abs(mdpe)) + (1-mdape) + ppe10 + ppe20
    return score

#The make_scorer function is required to turn this into a function the Gridsearch can use
from sklearn.metrics import make_scorer
scorer = make_scorer(my_scorer)

# Set of possible hyperparameters for the model to be searched through

param_grid = {
    'colsample_bytree':[0.4,0.6,0.8],
    'gamma':[0,0.03,0.1],
    'min_child_weight':[1.5,6],
    'learning_rate':[0.1,0.07],
    'max_depth':[1,3],
    'n_estimators':[5000],
    'reg_alpha':[1e-5, 1e-2, 0.75],
    'reg_lambda':[1e-5, 1e-2, 0.45],
    'subsample':[0.6,0.95]  
}

# Performing the gridsearch
xgb = xgboost.XGBRegressor()
cv = KFold(n_splits = 4,shuffle = True)
xgb_gs = GridSearchCV(xgb,param_grid,cv = cv,scoring = scorer, verbose = 1, n_jobs = -1)
xgb_gs.fit(X,y)

#The n_estimators in the gridsearch cv was set low in order to save processing time
# Manually increasing it for better accuracy for the final predictions
kwargs = xgb_gs.best_params_
kwargs['n_estimators'] = 20000

xgb = xgboost.XGBRegressor(**kwargs)
print(type(xgb).__name__)
ypred = xgb.fit(X,y).predict(X)

ypred_true = ypred*house['Price_Index'] # Predicted prices need to be scaled back up
evaluate(y_true,ypred_true)
my_scorer(y_true,ypred_true)

output = pd.DataFrame()
output['id'] = house['id'].values
output['sold_price'] = y_true
output['predicted_price'] = ypred_true

output.to_csv('output.csv',index = False)

# Plotting feature importance and exporting to a pdf
plot_importance(xgb)
plt.rcParams["figure.figsize"] = (25,12)
plt.savefig('Feature_Importance.pdf')

