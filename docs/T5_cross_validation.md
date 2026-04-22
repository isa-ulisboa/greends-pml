
# Model Evaluation and hyper-parameter Tuning

<details markdown="block">
<summary> Combining transformers and estimators in a pipeline </summary>

## Combining transformers and estimators in a pipeline

In the previous notes where we discussed [sklearn pipelines](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T3_missing_data_categorical_scaling.md), the pipeline was created with `Pipeline`. There is, however an alternative that makes the code shorter is to use `make_pipeline` [see sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html). This is a shorthand for the `Pipeline constructor`; it does not require, and does not permit, naming the estimators. Instead, their names will be set to the lowercase of their types automatically. The following piece of code shows how to create a pipeline that scales the attributes and applies a logistic regression.

  ```
  pipe_lr = make_pipeline(StandardScaler(),
                          LogisticRegression())
  ```

The pipeline is then typically used in the following manner over train and test sets:

  ```
  pipe_lr.fit(X_train, y_train)
  y_pred = pipe_lr.predict(X_test)
  train_accuracy = pipe_lr.score(X_train, y_train) # accuracy estimate over the same data used for training
  test_accuracy = pipe_lr.score(X_test, y_test) # accuracy estimate over an independent test set
  ```
---

</details>

<details markdown="block">
<summary> Using k-fold cross-validation to assess model performance </summary>

## Using k-fold cross-validation to assess model performance

The approach described above leads  in general to overfitting towards the train data set, and a bad performance over new examples. To prevent this, two diferent approaches can be followed

### The holdout method

<img src="https://github.com/isa-ulisboa/greends-pml/blob/main/docs/holdout_method_fig62.png" alt="Alt Text" width="500" >

### Cross-validation

<img src="https://github.com/isa-ulisboa/greends-pml/blob/main/docs/kfold_validation_fig_63.png" alt="Alt Text" width="500" >

In `sklearn`, cross-validation can easily be applied with [`cross_val_score`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html). Then, we can replace  `pipe_lr.fit(X_train, y_train)` in the script above by something like

  ```
  scores = cross_val_score(estimator=pipe_lr, # estimator with fit method
                             X=X_train,
                             y=y_train,
                             cv=10, # number of folds
                             n_jobs=1) # numbers of processors used (-1 for all processors)
  ```

that returns an array of scores of the estimator for each run of the cross validation. The parameter `cv` can be used to indicate which cross validation scheme should be used. It could take for instance one of the following: 

- [KFold](https://scikit-learn.org/stable/modules/cross_validation.html#k-fold): divides all the samples in groups of samples, called folds. This is equivalent to just use, e.g., `cv=10`.
- [GroupKFold](https://scikit-learn.org/stable/modules/cross_validation.html#stratified-group-k-fold): this a variation of k-fold which ensures that the same group is not represented in both testing and training sets
- [StratifiedKFold](https://scikit-learn.org/stable/modules/cross_validation.html#stratified-k-fold): this is a variation of k-fold which returns stratified folds: each set contains approximately the same percentage of samples of each target class as the complete set.
- [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/cross_validation.html#stratified-group-k-fold): The idea is to try to preserve the distribution of classes in each split while keeping each group within a single split.
- [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html): Provides train/test indices to split time-ordered data, where other cross-validation methods are inappropriate, as they would lead to training on future data and evaluating on past data. 

For instance, the following code stratifies folds by the target class `y`. So, if for instance there are 100 examples of class 0 and 10 examples of class 1, then all folds get 20 examples from class 0 and 2 examples for class 1 (since `n_splits=5`).

  ```
  # model
  clf = DecisionTreeClassifier(max_depth=10)
  # cv strategy 
  skf = StratifiedKFold(n_splits=5)
  # fit and predict over the validation set
  results = cross_val_score(clf, X_train, y_train, cv=skf)
  ```

**Script** to read italian wine regions data from the UCI repository, and applies stratified croass validation to predict the region from the wine attributes: (https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/cross_val_score_stratifiedkfold.ipynb)

---

</details>

<details markdown="block">
<summary> Learning and validation curves </summary>


## Learning and validation curves

A [learning curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve) shows cross-validated training and test scores for different training set sizes.

A [validation curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html#sklearn.model_selection.validation_curve) determine training and test scores for varying parameter values. This is equivalent to grid search (see below) for a single parameter.

**Script** to read italian wine region data and create learning curve for a given classifier: (https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/wine_regions_learning_curve.ipynb)

---

</details>

<details markdown="block">
<summary>Tuning machine learning hyper-parameters via grid search </summary>

## Tuning machine learning hyper-parameters via grid search

One of the most critical steps in machine learning is tuning hyper-parameters of the model, e.g. `max_depth` for a decision tree. It is possible and recommended to search the hyper-parameter space for the best cross validation score. See [sklearn grid search section](https://scikit-learn.org/stable/modules/grid_search.html#grid-search).

A search consists of:
- an estimator: regressor or classifier such as `sklearn.tree.DecisionTreeClassifier()`;
- a parameter space such as `param_grid = [{'max_depth': [4,5,6,7]}]`;
- a method for searching or sampling candidates, such as `GridSearchCV` or `RandomizedSearchCV`;
- a cross-validation scheme such as `StratifiedKFold`; and
- a score function such as the defaults `sklearn.metrics.accuracy_score` for classification and `sklearn.metrics.r2_score` for regression.

The main methods are:
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#gridsearchcv) that performs an exhaustive search over specified parameter values for an estimator;
- [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV). In contrast to `GridSearchCV`, not all parameter values are tried out, but rather a fixed number of parameter settings is sampled from the specified distributions.
- Scikit-learn also provides the `HalvingGridSearchCV` and `HalvingRandomSearchCV` estimators that can be used to search a parameter space using successive halving.

**Script** to apply a randomized search over a random forest classifier for the Iris data set: (https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/iris_randomizedsearchCV.ipynb)

---

</details>



