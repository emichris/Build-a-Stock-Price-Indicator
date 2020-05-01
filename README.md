# [Build a Stock Price Indicator](https://emichris-stock-predictor.herokuapp.com/)
![Header](prediction_engine/static/wallstreet.jpg)
Photo by [Rick Tap](https://unsplash.com/@ricktap?utm_medium=referral&amp;utm_campaign=photographer-credit&amp;utm_content=creditBadge) on [Unsplash](https://unsplash.com/)

### Table of Contents

1. [Instroduction](#intro)
2. [Project Descriprtion](#description)
3. [File Descriptions](#files)
4. [Deploying](#deploy)
4. [Results](#results)
5. [Licensing and Acknowledgements](#licensing)


## Introduction <a name="intro"> </a>

Investment firms, hedge funds and even individuals have been using financial models to better understand market behavior and make profitable investments and trades. A wealth of information is available in the form of historical stock prices and company performance data, suitable for machine learning algorithms to process.


## Project Description <a name="description"> </a>

The objective of this project is to build a stock price predictor. A user is able to get stock price predictions after training the model live on historical data. The system predicts 'Adjusted Close Price'. The final application is deployed on heroku at [https://emichris-stock-predictor.herokuapp.com/](https://emichris-stock-predictor.herokuapp.com/)


## File Descriptions <a name="files"></a>

There are 3 notebooks available; each with names correspoding to the work being done in them: 
    1. Downloading stock data using yahoo finance api in `pandas_datareader` library
    2. Predicting stock price with n-historical data 
    3. Predicting stock price with traditional machine learning method
Markdown cells are used throughout to assist in walking through the thought process for individual steps.  
The folder `prediction_engine` contains the necessary files, `.py`, `.html`, `.pkl` files needed for the application to work. 


## Deploying <a name="deploy"></a>

+ Clone the project
```
$ git clone https://github.com/emichris/Build-a-Stock-Price-Indicator.git
```

+ Create a project environment (Anaconda Env Recommended)
```
$ conda create --name myenv # Assuming Anaconda is installed, See above
$ source activate myenv
```

+ Setup dependencies
```
$ pip install -r requirements.txt
```

+ Start the server
```
$ cd Build-a-Stock-Price-Indicator/prediction_engine
$ python prediction_engine.py
```
Once the app is running, go to http:0:0:0:5000 to view the live app. 

### How it works
Once the app is running, enter a valid stock symbol eg AMZN for Amazon, TSLA for Tesla. The app fetches real time data using yahoo finance api.


## Results<a name="results"></a>
The applciation was tested using the following stocks: GOOG, TSLA, AMZN and received `r2 scores` of `0.97, 0.95, and 0.95` respectively.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>
I want to thank [Udacity](https://unsplash.com/) for the opportunity to work on this project as part of my Data Nanodegree Program. MIT Standard License applies. 
