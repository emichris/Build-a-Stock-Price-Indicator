""" 
@Author: Christian Emiyah
@Date: April 28, 2020
"""

# Load packages
import os, csv, os.path
from flask import request, redirect
from flask import Flask, render_template
import plotly.graph_objs as go
import json, plotly

#import relevant libraries
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np, math
import pandas_datareader.data as web
from sklearn.metrics import mean_squared_error, r2_score; 
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pickle

app = Flask(__name__)

stocks = []
endDate = dt.datetime.today().strftime('%Y-%m-%d')

@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
    
@app.route("/")
def first_page():
    endDate = dt.datetime.today().strftime('%Y-%m-%d') # reset enddate limit when user returns to home page
    return render_template("index.html", endDate=endDate)

# Get stock data 
def get_stock_data(symbol, start=dt.datetime(2015, 1, 1), end=dt.datetime.today()):
    '''
    download stock data over from yahoo api form start date to end date
    input
        stock - String representing stock symbol eg APPL
        start - datetime object represent start date; default Jan 1, 2010
        end - datetime object represent end date; default: Jan 1, 2020
    output
        historical stock data pulled from yahoo finance stock api from start to end dates
    '''
    try:
        stockData = web.DataReader(symbol, 'yahoo', start, end)
    except:
        print("Could Not get stock values for "+symbol)
        stockData = None
    
    return stockData

#Convert Date column to datetime
def get_date_features(df):
    '''
    input: 
        df is dataframe of historical stock data where index is the date. 
    output:
        new_df with added features from the day.. 
    '''
    new_df = df[['Adj Close']]
    new_df = new_df.reset_index()
    new_df.columns = ['date','price']
    new_df.loc[:, 'date'] = pd.to_datetime(new_df['date'],format='%Y-%m-%d')
    new_df['year'] = new_df.date.dt.year
    new_df['month'] = new_df.date.dt.month
    new_df['day'] = new_df.date.dt.day
    
    return new_df

def split_train_test(df, cv_size = .2, test_size = .2):
    '''
    Function splits a dataframe into training, validation and test sets
    '''
    #Get sizes of each of the datasets
    num_cv = int(cv_size*len(df))
    num_test = int(test_size*len(df))
    num_train = len(df) - num_cv - num_test
    
    # Split into train, cv, and test
    train = df[:num_train]
    validation = df[num_train:num_train+num_cv]
    train_cv = df[:num_train+num_cv]
    test = df[num_train+num_cv:]
  
    return train, validation, train_cv, test

def get_knn_model(df, target):
    
    params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
    knn = neighbors.KNeighborsRegressor()
    model = GridSearchCV(knn, params, cv=5)
    x_train = df[['year', 'month', 'day']]
    y_train = df[target]
    
    #fit the model and make predictions
    model.fit(x_train, y_train)
    print('XGBoost Best parameters: ', model.best_params_)
    
    return model

def get_pred(model, dates):
    '''
    get model predictions for a given date or set of dates
    input: 
        model - Machine Learning Model
        dates - pandas dataframe must contain 'year', 'month' and 'date'
    '''
    pred_vals = None
    try:
        pred_vals = model.predict(dates[['year', 'month', 'day']])
    except:
        print("Unable to use dataframe, make sure dataframe contains 'year', 'month', 'day'")
        
    return pred_vals

def return_figure(df, stock, r2score):
    """
    Args:
        df - dataframe with price and predicted price columns
        stock - name of the stock
        r2score - module performance score
    Returns:
        returns a plotly figure with price and predicted price line graph

    """

    graph_one = []

    graph_one.append(
      go.Scatter(
      x = df['date'],
      y = df['price'],
      mode = 'lines',
      name = "Actual Price"
      )
    )
    
    graph_one.append(
      go.Scatter(
      x = df['date'],
      y = df['preds'],
      mode = 'lines',
      name = "Predicted Price"
      )
    )

    layout_one = dict(title = 'Model Performance for %s <br> R2 Score: %.2f'%(stock, r2score),
                xaxis = dict(title = 'Date'),
                yaxis = dict(title = 'Price'),   )

    return dict(data=graph_one, layout=layout_one)

@app.route("/plot" , methods = ['POST', 'GET'] )
def main():
    '''
    the function is loaded when user submits a list of stocks with start and end date from the home page
    outputs data needed to display module performance and option to query stock data. 
    '''
    
    if request.method == 'POST':
        query = request.form['stockname']
        stocks = [x.strip().upper() for x in query.split(',')]
        startdate = request.form['startdate']
        endDate = request.form['enddate']
        
        figures = []
        
        for stock in stocks: 
            print("Getting stock data for "+stock)
            
            # Check if user entered both start and end dates
            if (startdate!='' and endDate!=''): 
                df = get_stock_data(stock, startdate, endDate)
            else: # use the default values
                startdate = dt.datetime(2010, 1, 1)
                endDate = dt.datetime.today().strftime('%Y-%m-%d')
                df = get_stock_data(stock, startdate, endDate)
            df = get_date_features(df)

            print("Training model..")
            model = get_knn_model(df, 'price') #get model

            train, validation, train_cv, test = split_train_test(df)
            train_cv['preds'] = get_pred(model, train_cv)
            test['preds'] = get_pred(model, test)

            print("\nTraining set RMS-error: ", math.sqrt(mean_squared_error(train_cv['preds'], train_cv['price'])))
            print("Test set RMS-error: ", math.sqrt(mean_squared_error(test['preds'], test['price'])))
            print("\nTraining set R2-error: ", r2_score(train_cv['preds'], train_cv['price']))
            print("Test set R2-error: ", r2_score(test['preds'], test['price']))

            train_cv.append(test)
            #train_cv.to_csv("static/numbers.csv", index=False, encoding="ISO-8859-1")

            # Save model
            filename = stock+"_model.pkl"
            pickle.dump(model, open(filename, 'wb'))
            
            with open('stocks.pkl', 'wb') as fp: #save stock list 
                pickle.dump(stocks, fp)

            figures.append(return_figure(train_cv, stock, r2_score(train_cv['preds'], train_cv['price'])))

        # plot ids for the html id tag
        ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

        # Convert the plotly figures to JSON for javascript in html template
        figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template("plot.html", endDate=endDate,stocks = stocks, ids=ids, figuresJSON=figuresJSON)
    else: 
        return render_template("index.html")

@app.route("/predict" , methods = ['POST', 'GET'] )
def predict():
    '''
    This is the function loaded when the user submits list of stocks and date to predict stock price.
    '''
    
    def transform_date(date):
        '''
        function converts a string date object into a dataframe 
        that can be used as input for the get_pred(model, date) function.
        '''
        
        new_df = pd.DataFrame({'date':pd.to_datetime(date)}, index=[1])
        new_df['year'] = new_df.date.dt.year
        new_df['month'] = new_df.date.dt.month
        new_df['day'] = new_df.date.dt.day

        return new_df
    
    if request.method == 'POST':
        query = request.form.getlist('stock')
        date = request.form['date']
        
        with open ('stocks.pkl', 'rb') as fp: # read stock list
            stocks = pickle.load(fp)
        
        preds = []
        for stock in query:
            print("Loading saved model")
            model = joblib.load(stock+"_model.pkl") #classifier.pkl"

            preds.append(round(get_pred(model, transform_date(date))[0]))
       

    return render_template("result.html", date=date, endDate=endDate, stock_pred = zip(query, preds), query=query, stocks=stocks)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
