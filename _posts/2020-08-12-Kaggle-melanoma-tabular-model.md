---
title: "Melanoma Prediction Kaggle Contest - Part 2: Predictions From Patient Metadata"
date: 2020-08-12
tags: [kaggle, competition, gradient boosted trees, xgboost, smote, oversampling]
mathjax: true
classes: wide
---

If you would prefer to go straight to the code to learn about my approach to the competition please [go here](https://github.com/sjhatfield/kaggle-melanoma-2020). The specific notebook for the tabular models is [here](https://github.com/sjhatfield/kaggle-melanoma-2020/blob/master/notebooks/tabular.ipynb).

Part 1 where I introduce the contest and explore the data can be found [here](https://sjhatfield.github.io/2020-08-12-Kaggle-melanoma-contest-exploration).

## Idea

Before spending time working with the image data, a model is developed to make predicitoins only using the tabular data (metadata). It is not expected that this will perform particularly well but the tabular model can be emsembled with an image model to possibly give a little boost in the score. Plus these models are quick to train and give an opportunity to develop more on an intuition about the dataset.

## Data Preparation

The columns that may be used for prediciton with their type are:

* Sex (2 categories)
* Approximate age (numerical)
* Anatomical site on body (6 categories)
* Width of full resolution image (numerical)
* Height of full resolution image (numerical)

With this kind of data a gradient boosting method for classification can be a good bet. The most popular library for fitting these kind of models is [XGBoost](https://xgboost.readthedocs.io). The summary of gradient boosting is that they train many weak classifiers, most often decision trees, and then ensemble them together for better performance. A decision takes the form of a series of yes/no questions which result in a classification. See below for an example from the XGBoost package [website](https://xgboost.readthedocs.io/en/latest/tutorials/model.html).

<img src="{{ site.url }}{{ site.baseurl }}/images/kaggle-melanoma/decision_tree.jpeg" alt="An example of two decision trees.">

This kind of model requires numerical variables so we must encode the categorical factors using dummy variables. This and other stages in the preparation of the tabular data was done in a [Python file](https://github.com/sjhatfield/kaggle-melanoma-2020/blob/master/src/data/prepare_data.py) which could be neatly imported into the model training [notebook](https://github.com/sjhatfield/kaggle-melanoma-2020/blob/master/notebooks/tabular.ipynb).

## Model Validation

As noted in the previous post, many patients had multiple entries in the training set. It was essential to make sure that patients did not exist in both training data and validation data. Fortunately, a contestant on Kaggle, [Chris Deotte](http://www.ccom.ucsd.edu/~cdeotte/), graciously provided data sets where not only had he provided cross-validation folds splitting the patients. But also made sure that each fold had the same proportion of malignant cases and balanced patient counts. This is referred to as **triple stratified leak-free** k-fold cross validation.

In the notebooks this can be seen in the extra column in all dataframes called "tfrecord". It is an integer from 0 to 15 so that folds may be created by randomly splitting those integers into 1, 3 or 5 non-intersecting subsets.

## The Model Pipeline

As mentioned in the previous post, there is extreme imbalance in the data where only $$1.76\%$$ of data entries are malignant. To combat this a sampling strategy must be employed. To begin with random over sampling was used from the imblearn package. This strategy simply resamples the minority class until it makes up a chosen proportion of training data. 

To search for good parameters for the sampler and model scikit-learns RandomizedSearchCV was used so that 100 different combinations of parameters were assessed using cross-validation. Here is the code creating the pipeline, parameters to be searched and the randomized search grid.

```python
model_randoversamp = Pipeline([
    ('sampler', RandomOverSampler(random_state=SEED)),
    ('classification', XGBClassifier(verbosity=1, random_state=SEED))
])

params_randoversamp = {
    'sampler__sampling_strategy': [0.1, 0.3, 0.5],
    'classification__eta': [0.0001, 0.001, 0.01, 0.1, 1],
    'classification__gamma': [0, 1, 2, 3, 4, 5],
    'classification__max_depth': [x for x in range(1, 11)],
    'classification__num_estimators': [100, 300, 500, 700, 900]
}

grid_randoversamp = RandomizedSearchCV(estimator=model_randoversamp, 
                          param_distributions=params_randoversamp, 
                          n_iter=100, 
                          scoring='roc_auc', 
                          cv=cv_iterator_int, 
                          verbose=1, 
                          n_jobs=-1)

```

### Results

This took 10.1 minutes to train on my local machine and the best performing set of parameters achieved a validation AUC-ROC of $$0.83$$ which was better than I expected. This model had a sampling strategy of $$0.3$$ meaning the malignant cases were resampled to bring their proporiton up to $$30\%$$ of the training data. The eta parameter used was $$0.1$$. This is a regularization constraint for which larger values shrink the weights of new features added to trees. The gamma paramter chosen was 4, which is the minimum loss decrease for a further partition to be made in a decision tree. The maximum depth of the trees chosnen was 2 and this is the maximum number of decisions until classification in the trees. Finally, the number of decision trees used was 100.

When this classifier was used on the test data the classifier achieved a score of $$0.7981$$ on the public Kaggle leaderboard. Impressive for a purely tabular model in my opinion.

## A Different Sampling Strategy

The next model used the same XGBoost classifier but used a more complex sampler to combat the imbalanced data.

```python
model_smote = Pipeline([
    ('sampler', SMOTE(random_state=SEED, n_jobs=-1)),
    ('classification', XGBClassifier(verbosity=1, random_state=SEED))
])

params_smote = {
    'sampler__k_neighbors': [1, 3, 5, 7],
    'classification__eta': [0.0001, 0.001, 0.01, 0.1, 1, 10],
    'classification__gamma': [0, 1, 2, 3, 4, 5],
    'classification__max_depth': [1, 2, 3, 4, 5, 6]
}
```

SMOTE stands for synthetic minority oversampling technique. This technique randomly selects a minority (malignant) data point and then finds $$k$$ of its nearest neighbors. One of those neighbors is randomly chosen and then a new datapoint is created at a random point between the two. This new data point is a **synthetic** example to be fed to the model. These additional synthetic points force the decision trees to expand their decision regions which may classify the unseen validation data better.

An obvious issue is that when the minority and majority classes overlap significantly (are not separable) then these new synthetic datapoints may not be representative of the minority class.

The randomized search took longer this time as SMOTE sampling is more computationally expensive and as it happens, had a lower best AUC-ROC score of $$0.8219$$ and $$0.7900$$ on the Kaggle public leaderboard.

## More Data

In an effort to improve the original randomly over-sampled model, the 2019 tabular data was added to the training data. Importantly, this external data was **not included** in any validation folds, only the training folds. This is because performance on external data should not be used to assess model performance. This was done by

```python 
tf_int = train_internal['tfrecord']
tf_ext = train_external['tfrecord']
tf_ext += 20
tf = pd.concat([tf_int, tf_ext], axis=0, ignore_index=True)

cv_iterator_ext = []
skf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for i, (idxT,idxV) in enumerate(skf.split(np.arange(15))):
    cv_iterator_ext.append((tf.isin(idxT) | (tf >= 20), 
                        tf.isin(idxV)))
```

This performed slightly worse than the model using just the internal data.

## Parameter Refining

In a final effort to squeeze some increased performance out of the classifier another parameter search was used but rather than using lists, probability distributions were used, with means equal to the chosen parameter from the original model. Here are the chosen parameter and the associated distribution:

* Sampling proportion 0.3 Uniform(low=0.1, high=0.5)
* Eta (regularization) 0.1 Gamma(shape=2, scale=0.05)
* Gamma (minimum loss) 4 Gamma(shape=2, scale=2)
* Maximum depth of tree 2 RandomInt(low=1, high=3)
* Number of estimators 100 RandomInt(low=50, high=150)

It should be fairly clear why the uniform and random integer distributuions were used. The gamma distribution was chosen for eta and gamma, as these variables are continuous and may take values in the range `$$[0, \inf)$$` which is the same as the gamma distributuion. It looks like a skewed normal distribution which cannot go negative which is appropriate for these varialbes. The mean of a gamma distribution is shape `$$\times$$` scale. 

For example, a random sample of $$25$$ values from Gamma(shape=2, scale=2) gave: 4.16200782, 4.78013484, 2.78268383, 3.52807528, 7.60847632, 8.69564004, 6.1258422 , 2.89771016, 2.10044363, 0.44427182, 5.0764685 , 3.87984299, 1.72199083, 4.773591  , 3.51366612, 3.63235669, 5.74539046, 2.63842483, 3.32101361, 0.69189948, 2.02039114, 9.40518265, 7.79320054, 2.70690933, 6.23665766.

It took 7.3 minutes to train this model and its score on the validation data was $$0.8310$$ which suggests better performance. Indeed this model did better on the Kaggle leaderboard scoring $$0.8176$$.

## Insight From The Model

Another great feature of gradient boosted trees, is that we may examine which variables were used most often to perform feature selection. Here is a plot of feature importance:

<img src="{{ site.url }}{{ site.baseurl }}/images/kaggle-melanoma/feature_importance.png" alt="Feature importance bar chart for the final tabular model.">

The final model had maximum depth $$1$$, so the this bar plot shows which of the $$142$$ estimators was based on each variable. The height and width of the original full resolution image were the best predictors, with age being the third best.

Here are some example trees:
<div class="row">
<div class="column">
<img src="{{ site.url }}{{ site.baseurl }}/images/kaggle-melanoma/tree_10.png" alt="Tree number 10.">
<img src="{{ site.url }}{{ site.baseurl }}/images/kaggle-melanoma/tree_29.png" alt="Tree number 29.">
</div>

<div class="column">
<img src="{{ site.url }}{{ site.baseurl }}/images/kaggle-melanoma/tree_99.png" alt="Tree number 99.">
<img src="{{ site.url }}{{ site.baseurl }}/images/kaggle-melanoma/tree_138.png" alt="Tree number 138.">
</div>
</div>



The leaf values at the bottom will be summed across all $$142$$ trees and then transformed into a probability using a sigmoid function. The first image indicates that if a patient is below 72.5 years old then then this makes it more likely the patient will be predicted as benign.

## What Have I Learnt?

Here is a list of the techniques, technologies, concepts completing this project has introduced to me for the first time.

* Randomized grid search over probability distributions in scikit-learn
* Using an XGBoost classifier to perform feature selection
* The suprising ability for solely tabular patient data to predict melanoma
* Different sampling techniques using the imblean library