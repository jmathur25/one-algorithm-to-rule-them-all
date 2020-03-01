One Algorithm to Rule Them All
===========================================

Description
===========
There is an implicit paradigm when it comes to machine learning classification: first try a variety of models (Logistic
Regression, Support Vector Machine, Random Forest, Gradient
Boosting, Artificial Neural Nets, etc.), and depending on your criteria
(accuracy, speed, need for interpretability, etc.) you can pick one.

This can feel very unsatisfying - it feels as if there should be a way to know beforehand which models are more suitable for a dataset. Consequently, we tried to make an algorithm that can identify whether Logistic Regression or Random Forest will perform better prior to any model fitting.

-`one-algorithm-to-rule-them-all.pdf` contains our presentation and an overview of our work.

-`metrics.py` contains the metrics that we use to decide which model is better.

-`sklearn_datasets.ipynb` will work out of box

-`airlines.ipynb` and `hotel_bookings.ipynb` need you to download data from Kaggle.
