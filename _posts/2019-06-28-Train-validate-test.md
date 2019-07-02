---
title: "Splitting Data Into Training, Validation and Testing Sets"
date: 2019-06-28
tags: [concepts, analytics, model selection]
mathjax: true
classes: wide
---

## Why Split At All?

After training a model the first question someone is bound to ask is "So how good is it?". To answer that question you need to assess the models quality in an **unbiased**,  way. In general, this means you need to use the model to predict or classify from data that **it has not been trained on**. Let me give a slightly extreme example, that highlights why assessing a model on the data it has been trained on leads to an incorrect assessment of it's quality.

### Example

In my time as a teacher I have had many classes and spent hours looking at attendence lists. Every now and again, I would notice a class that had a skewed distribution of last names according to the alphabet. In fact recently, I had a class of 12 students whose last names were distributed like so[^1]:

Chaudri, Chin, Gardener, Radley, Smith, Simon, Soo, Subram, Sydney, Truman, Yao, Zao.

As you may notice, most of the names are in the second half of the alphabet. By the time I had gotten to the fourth name taking attendence I was already to R. A model to predict the first letter, of the last name of mathematics students which always predict **S** would perform well on this dataset. By using a letters distance from the prediction in the alphabet as the error, it's RMSE[^2] would only be 7.87.

However, do we really think that always predicting a students name to begin with an S will perform well in general. It is debatable whether last names will be evenly distributed accross the alphabet but in a school of many nationalities like mine was it may be fairly even. Surely, predicting a last name to begin with one of the median letters (M or N) in the alphabet would lead to better accuracy. On a more evenly distributed class:

Adams, Chad, Dae-Jung, Frank, Iban, Issiah, Lee, Oswald, Ridler, Umaya, Vahit, Xi.

The S-predictor has an RMSE of 10.32 whilst a more sensible prediction of M has an RMSE of 7.38.

Whilst this is a slightly unrealistic example, it shows that a model can (and most likely will) fit the biases found in the dataset. This is called **overfitting**.

## Assessing The Quality Of A Single Model

If a single model is to be assessed then the dataset should be split into two parts, a training set and a testing set. The model should be fit to the training data and then the testing set should be used to assess it's quality. This usually means generating responses to the training sets predictor variables and then assessing their difference to the actual responses. Generally, it is recommended to select 70-90% for training and 10-30% for testing.

However, this approach does not work if we are fitting multiple models, picking the best and then assessing the quality of the final one chosen.

## How To Pick A Model From Many And Then Assess Quality

The above method of splitting data into training and testing is not sufficient when we are picking the best model from many. This is because if we were to fit all the models on the training data then pick the best to perform on the test data we would inadvertently pick the one that matches the biases of the testing data. For example, if we think about guessing the first letter of a students name in a class and the test dataset happens to be the unusually distributed one that I taught then we are going to choose the model which predicts more towards the second half of the alphabet.

To avoid this pitfull, a third dataset needs to be constructed. We need a training set to fit the models, a validation set to select the best performing model and a final test dataset to assess the quality of the final model chosen. Generally, it is recommened to select 50-70% for training and then split the remaining data evenly between validation and testing.

## The Importance Of Choosing Unbiased Validation And Testing Sets

As the example above demonstrated, it is essential that validation and testing sets are chosen that represent the overall dataset well and are not biased. Imagine working with hamburger sales data for a restaurant and your model was fit to the weekday data and then tested on the weekend data. Your model would most likely perform very badly because the sales data will be significantly different during the week compared to the weekend.

To avoid this issue, usually it is best to randomly select datapoints. One way of doing this is to shuffle the rows of the dataframe you are working with and then select the first $$x%$$ of rows. In R this would look like

```r
dataset[sample(nrow(dataset)),]
```

and in Python using Pandas

```python
df.sample(frac=1, inplace=True)
```

It can be important to make sure a certain proportion of responses are in the validation and test sets. For example, if you are building a model to predict whether bank customers default on a loan and 10% of your dataset default. Then it could be important to make sure 10% of the validation and test sets defaulted also. This can be done by randomly selecting and then checking whether the proportion is correct or splitting the dataset into the defaulters and non-defaulters and selecting proportions of each to be then combined again.

[^1]: These names are not real but their distribution in the alphabet is tyhe same as a class I taught.

[^2]: [Root-mean-square error](https://www.wikiwand.com/en/Root-mean-square_deviation).