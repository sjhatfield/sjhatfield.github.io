---
title: "Used Car Purchase Helper - A Machine Learning Powered Application"
date: 2020-05-20
tags: [projects, machine learning, regression]
mathjax: true
classes: wide
---

Spoiler: if you want to skip to the finished product follow [this link](http://178.79.141.9/). I have not purchased a domain name and apologise that I have not set up HTTPS so your browser may complain that the site is dangerous. If you would prefer to learn about this project by reading the code [go here](https://github.com/sjhatfield/car-purchase-help).

## The Problem

Buying a used car is a tricky task where a wrong decision can have large financial implications. There are hundreds of options of make and model available and then possibly thousands of options once a make and model have been chosen. Vehicles vary vastly in condition, age and price, and it is impossible to know you have made the best decision when purchasing a used car. The client is in the position of wanting to purchase a used vehicle but is unsure of what would be a good deal, which model would be the best for their needs and how long they would expect the vechicle to last once purchased. This application was developed to help the client assess how good a deal an advert for a used car is.

## The Ideal Outcome

The application will be a success if the client is able to input the details of a used car they have found for sale and the application is able to instantly give them feedback as to whether the listing appears to be a good deal. Ideally, the application will be able to give them more than just a yes/no on whether the deal appears to be good. Possibly the application will even be able to provide the same information about similar vehicles.

### Formulation as a Machine Learning Problem

The purpose of this project is for the writer to complete their first full, end-to-end, machine learning powered application. Therefore, the model used will **not** be a complex one, as the focus of this project is to go all the way from a problem to a functioning solution. This project follows the structure outlined in Emmanuel Ameisen's wonderful book [Building Machine Learning Powered Applications](https://www.oreilly.com/library/view/building-machine-learning/9781492045106/). Once this project has been completed the writer is going to build more applications, using more complex models. **This project will simply use a linear regression to predict the price of a vehicle and give the user advice based on that value**.

## Data Collection and Analysis

A dataset was found on [Kaggle](https://www.kaggle.com/austinreese/craigslist-carstrucks-data/), containing over 500,000 scraped Craigslist car listings in the US. Importantly, the dataset contains entries for many important features of a vehicle. Most importantly it gives, `price`, `mileage`, `year`, `manufacturer`, `model` and `condition`. This dataset has some obvious drawbacks which will be discussed later but it is large enough to gain insight from and will be used in the project. Many thanks to [Austin Reece](https://www.kaggle.com/austinreese) for providing it.

Of course, the first thing to do is spend some time examining the dataset and assessing its quality and viability. For an exploratory analysis please see [this notebook](https://github.com/sjhatfield/car-purchase-help/blob/master/notebooks/initial_exploration.ipynb).

The important conclusions of the intial exploration are that the dataset is certainly viable to gain insight from and important features have outliers that must be removed.

## Data Formatting

It is good practice to treat the original dataset as immutable and use a function to format it into a usable form every time it is loaded. [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) is a project template which was used for this project and [here](https://drivendata.github.io/cookiecutter-data-science/#data-is-immutable) they give some reasons why it is best practice to treat the data as immutable.

Functions to format the data were written in a [`data_processing.py`](https://github.com/sjhatfield/car-purchase-help/blob/master/car_purchase_help/data_processing.py), kept separate from the notebooks, as notebooks are used for **exploration** and **communication** only. These functions can then be imported into any python file or notebook where they are needed.

The main function `format_raw_df` takes in the Pandas dataframe loaded from the csv file and performs the processing steps which were determined through the initial exploration. Only one column, `county`, was dropped from the DataFrame as it was $100%$ missing. It was tempting at this point to greatly reduce the size of the DataFrame by dropping all other columns that did not seem useful. However, machine-learning solutions should be iterated upon and improved over and over, and it is impossible to say whether columns would be used in later solutions.

As shown by the exploration there were outliers in important columns so these were removed. It was decided to remove any vehicle produced before 1981 and after 2019 (there were some cars from the future). Also, milage values below $1,000$ and above $300,000$ were removed, as well as price below $\$500$.

## A Second Exploration After Formatting

It was important to assess the impact the formatting had on the data before building a model. If the data processing steps had altered the districution of data or removed so many points prediciton was impossible it would be a problem. Therefore, a second exploration took place after formatting in [this notebook](https://github.com/sjhatfield/car-purchase-help/blob/master/notebooks/exploration_after_cleaning.ipynb).

The same plots as the initial exploration were produced and the shapes of distributions and correlations were determined to be the same as before. For example, the distribution of year of manufacture was unchanged.

<img src="{{ site.url }}{{ site.baseurl }}/images/car-purchase-help/year_manufacture_histogram.png" alt="Histogram showing yer of manufacture. Why do you think there a dip at 2008-2009?">

However, another set of outliers was discovered which would be very influential in making the all-important prediciton of price of vehicle. The plot below shows a Honda, CR-V from 2013 listed as having a price below $\$2,500$ which is vastly different to the other prices for this vehicle, even more so when you consider the fact that it has less than $20,000$ miles on the clock. These outliers could pull down the prediciton of the value of this vehicle. Therefore, it was decided they should be removed and this will be done when models are trained.

<img src="{{ site.url }}{{ site.baseurl }}/images/car-purchase-help/honda_crv_2013.png" alt="Scatter plot of mileage against price for Honda CR-Vs from 2013.">

## Training Models

As outlined earlier, the purpose of this project was not to develop fancy, sophisticated models, it was to complete a full machine learning project from start to finish. Therefore, simple linear regression was used on specific vehicles (make and model) from each year. From the mileage, price was predicted which according to some quick research is an appropriate way to predict price. Regressions were fit and saved for any vehicle with 30 or more data points after outliers were removed, according to the $1.5 IQR +/- Q_3/Q_1$ rule. The models were pickled and saved in the model directory for quick access by the web application. [This notebook](https://github.com/sjhatfield/car-purchase-help/blob/master/notebooks/model.ipynb) did the training using the functions from the [`model1.py` file](https://github.com/sjhatfield/car-purchase-help/blob/master/car_purchase_help/model1.py).