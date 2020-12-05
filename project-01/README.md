
# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the [Udacity Azure ML Nanodegree](https://www.udacity.com/course/machine-learning-engineer-for-microsoft-azure-nanodegree--nd00333).
In this project, we build and optimize an **Azure** ML pipeline using the Python SDK, **Hyperdrive**, Scikit-learn and **AutoML**.

We learn the best parameters on a given sklearn model using Hyperdrive and compare this model to the best models found using AutoML.

## Summary
The dataset contains information about customers or potential customers of a bank. The columns include personal information like age,job,marital status and education level and financial information like housing status, employment status and loans and information about previous contact with the person.

We seek to predict the likeliness of acquisition of a person as a customer.

First we optimized hyper-parameters using hyperdrive, then we trained multiple models using automl to retreive a best performing model.

The hyperdrive model achieved an accuracy of 0.9082, the best automl model 0.9177. We can see that the AutoML model has performed better.


## Scikit-learn Pipeline
We perform the following steps to learn the parameters of the scikit-learn model:
1. Download dataset
2. Clean dataset
3. Split dataset intro train & test
4. Train model with logistic regression and the train split
5. Test accuracy with trained model and the test split
6. Repeat step 4 and 5 for each set of hyperparameters
7. Choose hyperparameters that lead to the best performing model

**What are the benefits of the parameter sampler you chose?**

I chose the `RandomParameterSampling` method ([documented here](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling?view=azure-ml-py)) because i wanted to give each possible value the same chance to be chosen:

> Random sampling supports discrete and continuous hyperparameters. It supports early termination of low-performance runs. Some users do an initial search with random sampling and then refine the search space to improve results.
>
> In random sampling, hyperparameter values are randomly selected from the defined search space.

([source](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters))

This way i do not introduce any bias with the sampling.

I configured the sampler to uniformely choose the regularization parameter  ``--C`` between 0.005 and 100 and the number of iterations to be one of the following values: {20, 40, 50, 80, 100, 120, 150, 180}. Smaller iteration numbers would lead to failure of converging in the logistic regression.


**What are the benefits of the early stopping policy you chose?**

I chose the `BanditPolicy` method ([documented here](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py)) because it allows to terminate poorly performing runs where iterations do not lead to big increases in accuracy. This can safe ressources and time, as testing all possible hyperparameter combinations would be very expensive.

> Bandit policy is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.

([source](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters))

In our case this means that runs will be stopped when the accuracy is not within the specified distance to the best performing run.

## AutoML
I used AutoML to find the best performing model across different types of classificators regarding accuracy in threefold cross validation.

AutoML stands for "Automatic Machine Learning". In AutoML, different Model Architectures and Networks are consequently trained and tested on different metrics. 
In the end, the model with the highest score in accuracy is chosen as the best model. Accuracy is the percent of predicted labels that exactly match the true labels.	

According to the AutoML run of this project, the best model is a [pre-fitted soft voting classifier](https://docs.microsoft.com/en-us/python/api/azureml-automl-runtime/azureml.automl.runtime.shared.model_wrappers.prefittedsoftvotingclassifier?view=azure-ml-py).

It is used in a pipeline after a general data transformer, which is configured to do no major changes to the input data.

A soft voting (also called majority rule) classifier chooses the class based on the outputs of multiple sub-classifiers. As the name suggest, it chooses the class outputted by the majority of classificators. ([source](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html))

The soft voting classifier of this run consists of multiple XGBoostClassifiers (parallel gradient tree boosting), a LightGBMClassifier (Light Gradient Boosting Machine) and a SGDClassifierWrapper (stochastic gradient descent). In the file [`best_automl_estimators.txt`](best_automl_estimators.txt) all estimators and their parameters are mentioned.

## Pipeline comparison
The hyperdrive model achieved an accuracy of 0.9082, the best automl model 0.9177. We can see that the AutoML model has performed better. 

The better performance is most likely due to the more complex pipeline-style architecture of the soft-voting classificator found by AutoML compared to the logistic regression model used with hyperdrive.

While our Hyperdrive run used a logistic regression, which is a very basic mathematic method, AutoML chose a soft voting classifier consisting mostly of gradient tree bosting estimators. Gradient boosting methods produce ensemble-based models that consist of decision trees. In multiple stages the model gets build and generalized to allow optimization of arbitrary differentiable loss functions.  ([source](https://www.wikiwand.com/en/Gradient_boosting#/overview)) 

## Future work
There are multiple options to further improve these experiments.

If we try out other models (instead of logistic regression) for hyperdrive, we can get a more exhaustive comparison between AutoML and Hyperdrive. It would be especially interesting to use a gradient boosting method like XGBoost for Hyperdrive, as this is the base technique for the model found using AutoML.

If we increase the hyperparameter search space, hyperparameter optimization will take more time, but also more options will be included and out-of-scope unexpected results can occur that might lead to higher overall accuracy of the tuned model.

Increasing the AutoML timeout was not possible on the timed VMs provided by Udacity, but it would allow AutoML to try out more different models and maybe find a better final model in the end. It will increase our waiting time though as well.

We can also improve our data preparation. Currently, only very few cleaning steps are done in the notebook. Many successfull projects use for example feature engineering, which means combining multiple existing features to create new ones, most likely by using domain-knowledge, that helps us to find useful calculations. If for example a speed and a duration are giving in a dataset, it might make sense to calculate the distance by multiplication of these values. Results will highly vary by the used models, so there is no guarantee for improvement, but it will be worth a try, especially because AutoML tries out multiple models.
