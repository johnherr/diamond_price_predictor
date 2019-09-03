import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import (mean_absolute_error, r2_score)


def read_data(filepath):
    '''Reads text diamond data into pandas DF, and sets appropriate datatypes
    Categorical datatypes are ordered appropriately to improve information gain
    when splitting using forest models.
    '''
    df = pd.read_csv('diamonds.txt',delimiter='\t', dtype={'cut':'category','color':'category',
                                                       'clarity':'category'})
    
    df['cut'] = df['cut'].cat.reorder_categories(['Fair', 'Good', 'Very Good',
                                                  'Premium', 'Ideal'])
    df['color'] = df['color'].cat.reorder_categories(['J', 'I', 'H', 'G', 'F', 
                                                      'E', 'D'])
    df['clarity'] = df['clarity'].cat.reorder_categories(['I1', 'SI2', 'SI1', 
                                                          'VS2', 'VS1', 'VVS2', 
                                                          'VVS1', 'IF'])
    return df

def train_test(data, model_filepath='trained_model.joblib', test_size=.2, random_state=42):
    '''Splits data into training and test sets.
    Model generated using train data and evaluated with test data
    ARGS:
        filepath: filepath to raw diamond data
    RETURNS:
        ....
    '''
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state)
    
    train(train_data, model_filepath)
    predict(test_data, model_filepath)
    
def train(data, filepath='trained_model.joblib', model_type='boosting'):
    '''Generates and saves a model using the train dataset. 
    ARGS:
        data: pandas df for training
        filepath: filepath to save the trained model
    '''
    
    X_train, y_train = process(data)
    y_train = np.log(y_train) # Training with price logorithmic scale
    
    if model_type == 'boosting':
        model = xgb.XGBRegressor(max_depth=5, learning_rate=0.2, n_estimators=200, silent=True,
                             objective='reg:linear', booster='gbtree', n_jobs=4, nthread=None,
                             gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,
                             colsample_bytree=1, colsample_bylevel=1, reg_alpha=0,
                             reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0,
                             seed=None, missing=None)
    else:
        model = RandomForestRegressor(n_estimators =10)

          
    model.fit(X_train, y_train)
    dump(model, filepath) 
    
def predict(data, filepath='trained_model.joblib', output='predictions.txt'):
    '''Predicts prices for the test dataset. 
    ARGS:
        data: pandas df for testing
        filepath: filepath to load the trained model
        output: filepath of CSV with predicted prices 
    '''
    X_test, y_test = process(data)
    model = load(filepath)
    predictions = np.exp(model.predict(X_test)) # Needed to convert back from log scale
    np.savetxt('predictions.csv', predictions, delimiter="\t")
    score(predictions, y_test, model)

def process(df):
    '''Preprocessing of data before training or testing.
    Processing includes:
        - Converting to categorical type & ordering
        - Generating new features 'log_price', 'surface area'
        - Replacing zero values with means
        - Dropping unused features 'x','y','z'
    ARGS:
        df:pandas dataframe
    RETURNS:
        processed dataframe'''
    
    processed_df = df.copy()
    
    #Replace 0s witgh mean values for x,y,z
    dims =['x','y','z']
    for _ in dims:
        processed_df[_].replace(to_replace=0, value=np.mean(df[_]), inplace=True)

    processed_df['cut'] = df['cut'].cat.codes
    processed_df['color'] = df['color'].cat.codes
    processed_df['clarity'] = df['clarity'].cat.codes
    
    # Dabbling with Feature Engineering
    #processed_df['bounding_volume'] = (processed_df['x'] * processed_df['y'] * processed_df['z'])
    #processed_df['surface_area'] = (processed_df['x'] * processed_df['y'])
    #processed_df = processed_df.drop(['x','y','z'], axis = 1)
    #processed_df['ccc'] = (processed_df['cut']*processed_df['color']*processed_df['clarity'])**2
   

    y = processed_df['price']
    X = processed_df.drop(['price'], axis=1)
    
    
    return X, y

def score(y_predict,y_test, model):
    '''
    Calculates MAE and R2 as performance metrics
    Also generates plots of residuals
    '''
    residuals = y_predict-y_test
    print('Mean Absolute Error:', np.mean(np.abs(residuals)))
    print('R^2: ', r2_score(y_predict, y_test))
    plt.rcParams.update({'font.size': 19})
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].scatter(y_test, residuals, s=2, alpha=.6)
    ax[0].set(xlabel='True Price, $', ylabel='Residual, $')
    ax[1].scatter(y_test, residuals/y_test, s=2, alpha=.6)
    ax[1].set(xlabel='True Price, $', ylabel='Residual / True Price')
    plt.tight_layout()
    fig.savefig('images/residuals.png', dpi=100)
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
    plot_importance(model, ax2)
    plt.tight_layout()
    fig2.savefig('images/feature_importance.png', dpi=100)
   
if __name__ == '__main__':
    ap = argparse.ArgumentParser('Train or use a model for predicting diamond prices')
    ap.add_argument('mode', help='\'train\', \'predict\', or \'train_test\'')
    ap.add_argument('-i', '--input_file', help='raw input file', default='diamonds.txt')
    ap.add_argument('-m', '--model', help='model_file', default='trained_model.joblib')
    ap.add_argument('-o', '--predictions_file', help='output of predicitons', default='predictions.txt')

    args = ap.parse_args()
    if args.mode not in ['train', 'predict', 'train_test']:
        print('Valid modes are \'train\', \'predict\', or \'train_test\' ')
    
    data = read_data(args.input_file)
    if args.mode == 'train':
        model = self.train(data)
    elif args.mode == 'predict':
        predictions = predict(data, args.model)
    elif args.mode == 'train_test':
        model = train_test(data)