# Build a Stock Price Indicator
Photo by Rick Tap on Unsplash
![Header](stock_price_prediction/static/wallstreet.jpg)
<a style="background-color:black;color:white;text-decoration:none;padding:4px 6px;font-family:-apple-system, BlinkMacSystemFont, &quot;San Francisco&quot;, &quot;Helvetica Neue&quot;, Helvetica, Ubuntu, Roboto, Noto, &quot;Segoe UI&quot;, Arial, sans-serif;font-size:12px;font-weight:bold;line-height:1.2;display:inline-block;border-radius:3px" href="https://unsplash.com/@ricktap?utm_medium=referral&amp;utm_campaign=photographer-credit&amp;utm_content=creditBadge" target="_blank" rel="noopener noreferrer" title="Download free do whatever you want high-resolution photos from Rick Tap"><span style="display:inline-block;padding:2px 3px"><svg xmlns="http://www.w3.org/2000/svg" style="height:12px;width:auto;position:relative;vertical-align:middle;top:-2px;fill:white" viewBox="0 0 32 32"><title>unsplash-logo</title><path d="M10 9V0h12v9H10zm12 5h10v18H0V14h10v9h12v-9z"></path></svg></span><span style="display:inline-block;padding:2px 3px">Rick Tap</span></a>



### Table of Contents

1. [Instroduction](#intro)
2. [Project Descriprtion](#descrption)
3. [File Descriptions](#files)
4. [Deploying](#deploy)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Introduction <a name="intro"> </a>
Investment firms, hedge funds and even individuals have been using financial models to better understand market behavior and make profitable investments and trades. A wealth of information is available in the form of historical stock prices and company performance data, suitable for machine learning algorithms to process.

## Project Description <a name="description"> </a>
The objective of this project is to a stock price predictor that takes daily trading data over a certain date range as input, and outputs projected estimates for given query dates. The system predicts the Adjusted Close Price. 

## File Descriptions <a name="files"></a>

There are 3 notebooks available here to showcase work related to the above questions.  Each of the notebooks is exploratory in searching through the data pertaining to the questions showcased by the notebook title.  Markdown cells were used to assist in walking through the thought process for individual steps.  

There is an additional `.py` file that runs the necessary code to obtain the final model used to predict salary.


## Deploying <a name="deploy"></a>

+ Clone the project
```
$ git clone https://github.com/emichris/.. 
```

+ Create a project environment (Anaconda Env Recommended)
```
$ conda create --name myenv # Assuming Anaconda is installed, See above
$ conda create -n pht python=3.7 anaconda 
$ source activate myenv
```

+ Setup dependencies
```
$ pip install -r REQUIREMENTS.txt
```

+ Start the server
```
$ cd Stock-Market-Prediction/stock_price_prediction/
$ python prophet.py
```

### How it works
Once the app is running, enter a valid stock symbol eg AMZN for Amazon, TSLA for TSLA. The app fetches real time data using yahoo finance api. Once in a while, an error comes in retrieving data from yahoo finance as they check for captcha to make sure no automated system is using their data. In that case, just go back to the homepage and try again.


## Results<a name="results"></a>
The applciation was tested using the following stocks: the values of the root mean  square error, mean absolute error on a subset testing test is given below:


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Stack Overflow for the data.  You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/stackoverflow/so-survey-2017/data).  Otherwise, feel free to use the code here as you would like! 