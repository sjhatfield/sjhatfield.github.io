---
title: "Splitting Data Into Training, Validation and Testing Sets"
date: 2019-06-28
tags: [concepts, analytics, model selection]
mathjax: true
classes: wide
---

## Why Split At All?

After training a model the first question someone is bound to ask is "So how good is it?". To answer that question you need to assess the models quality in an **unbiased,  way. In general, this means you need to use the model to predict or classify from data that **it has not been trained on**. Let me give a slightly extreme example, that highlights why assessing a model on the data it has been trained on leads to an incorrect assessment of it's quality.

### Example

In my time as a teacher I have had many classes and spent hours looking at attendence lists. Every now and again, I would notice a class that had a skewed distribution of last names according to the alphabet. In fact recently, I had a class of 12 students whose last names were distributed like so[^1]:

Chaudri, Chin, Gardener, Radley, Smith, Simon, Soo, Subram, Sydney, Truman, Yao, Zao.

As you may notice, most of the names are in the second half of the alphabet. By the time I had gotten to the fourth name taking attendence I was already to R. A model to predict the first letter, of the last name of mathematics students which always predict **S** would perform well on this dataset. By using a letters distance from the prediction in the alphabet as the error, it's RMSE[^2] would only be 7.87.

However, do we really think that always predicting a students name to begin with an S will perform well in general. It is debatable whether last names will be evenly distributed accross the alphabet but in a school of many nationalities like mine was it may be fairly even. Surely, predicting a last name to begin with one of the median letters (M or N) in the alphabet would lead to better accuracy. On a more evenly distributed class:

Adams, Chad, Dae-Jung, Frank, Iban, Issiah, Lee, Oswald, Ridler, Umaya, Vahit, Xi.

The S-predictor has an RMSE of 10.32 whilst a more sensible prediction of M has an RMSE of 7.38.

Whilst this is a slightly unrealistic example, it shows that a model can (and most likely will) fit the biases found in the dataset. This is called overfitting.

## How To Assess The Quality Of One Model

If a single model is to be assessed then the dataset should be split into two parts, a training set and a testing set. The model should be fit to the training data and then the testing set should be used to assess it's quality. This usually means generating responses to the training sets predictor variables and then assessing their difference to the actual responses.

However, this approach does not work if we are fitting multiple models, picking the best and then assessing the quality of the final one chosen.

## How To Pick A Model From Many And Then Assess Quality

[^1]: These names are not real but their distribution in the alphabet is accuracte to a class I taught.

[^2]: [Root-mean-square error](https://www.wikiwand.com/en/Root-mean-square_deviation).