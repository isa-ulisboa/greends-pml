# Ensemble methods

The goal of ensemble methods is to combine different classifiers into a meta-classifier that has better generalization performance than each individual classifier alone. 

<details markdown="block">
  <summary> Illustration of  the ensemble approach</summary>

  <img src="https://github.com/isa-ulisboa/greends-pml/blob/main/figures/ensemble_approach_fig_72.png" width="600" >

 </details>

The motivation for this can be found in the "Wisdom of the crowd" concept, which originates with observations of Francis Galton in 1906: see for instance this [BBC video](https://youtu.be/iOucwX7Z1HU?si=Dk1Tc4J-bv9Ow1rG).

---

<details markdown="block">
<summary> Random Forests </summary>

## Random forests

We have discussed and used  decision and regression trees: recall that the goal is to create a tree that minimizes the impurity (measured by entropy of by the Gini indez) of the new nodes.  -- see for instance [Normalized Nerd videos](https://www.youtube.com/channel/UC7Fs-Fdpe0I8GYg3lboEuXw) on classification and regression trees. 

Random forests are ensemble learning methods that involve:
  - (bootstraping) Creating a collection of trees from bootstrap samples (sampling with replacement); [see meaning](https://en.wikipedia.org/wiki/Bootstrapping)
  - (decorrelating) Decorrelate models by randomly selecting features
  - (aggregating) Ensembling the collection of trees by majority vote.

<details markdown="block">
<summary> Illustration of the construction of a random forest</summary>

<img src="https://github.com/isa-ulisboa/greends-pml/blob/main/figures/random_forests.png" width="600" >

</details>


<details markdown="block">
  <summary> Pseudo-code with the main steps to create a random forest. </summary>

  
  ### Step 1: Initialize Parameters
  1. Set the number of trees `N_trees`.
  2. Define the maximum depth of each tree `max_depth`.
  3. Set the number of features to consider when splitting `max_features`.
  
  ### Step 2: Prepare the Data
  1. Split the dataset into training and testing sets.
  2. Preprocess the data (e.g., handle missing values, normalize if needed).
  
  ### Step 3: Build the Random Forest
  1. Initialize an empty list `forest` to store decision trees.
  
  2. For each tree `i` in range(1, N_trees):
     - **Step 3.1:** Create a bootstrap sample:
       - Randomly sample the training data with replacement to create a subset.
     - **Step 3.2:** Train a decision tree:
       - Select a random subset of features (`max_features`).
       - Grow the tree using the bootstrap sample:
         - At each node, split on the best feature (based on criteria like Gini Impurity or Entropy for classification, or variance for regression).
         - Stop splitting if `max_depth` is reached or other stopping criteria are met.
     - **Step 3.3:** Add the trained decision tree to `forest`.
  
  ### Step 4: Make Predictions
  1. For a new data point:
     - Pass it through each tree in the forest.
     - Collect predictions from all trees (majority vote for classification, or weighted mean for regression).
  
  2. Return the final prediction.
  
  ### Step 5: Evaluate Performance
  1. Use the testing set to evaluate accuracy or other metrics (e.g., precision, recall).
  
  ---
  
  </details>
  
  <details markdown="block">
  <summary> Why do random forest reduce the variance of the estimator?</summary>
  
  
  For simplicity, let's consider regression trees and show that the goal of ensembling trees with random forests is reducing the variance. 
  
  Let  $X_i$ be  the random variable  that represents the predition for the regression tree $T_i$ from the collection, with $\rho={\rm cor}\[X_i,X_j\]$ being the correlation between $X_i$ and $X_j$. The prediction from the ensemble is
  
  $$\bar{X}=\frac{1}{n} \left( X_1+\dots+X_B \right)$$
  
  and its variance is given by
  
  $${\rm Var}[\bar{X}]=  \rho \\, \sigma^2 + \frac{1-\rho}{B} \sigma^2,$$
  
  where ${\rm Var}[X-i]=\sigma^2$ and $B$ is the number of bootstrap samples. **As long as $\rho$ does not grow with $B$**, which is why the trees are decorrelated, using a larger ensemble will increase $B$ and reduce ${\rm Var}[\bar{X}]$, which is the goal of ensembling estimators.
  
  ---
  
  </details>
  
  <details markdown="block">
  <summary> Script to create random forest with scikit-learn</summary>
  
  ```python
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.datasets import load_iris
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score
  
  # Load the Iris dataset
  iris = load_iris()
  X = iris.data
  y = iris.target
  
  # Split the dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  # Create a Random Forest classifier
  rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
  # Train the classifier
  rf_classifier.fit(X_train, y_train)
  # Make predictions on the test set
  y_pred = rf_classifier.predict(X_test)
  # Evaluate the accuracy of the classifier
  accuracy = accuracy_score(y_test, y_pred)
  print("Accuracy:", accuracy)
  ```

  </details>

  
  <details markdown="block">
  <summary> Script that illustrates that random forests are easily parallelizable for reducing computation time. </summary>
  
  The script uses the option `jobs=-1` to run `RandomForestClassifier` over all cores. Compare processing time for that same code on your local machine when setting `jobs=1` (using a single core). Random forests  are easily parallelizable since each tree is grown independently from the remainder trees.

  ```python
  from sklearn.datasets import make_classification
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score
  import time
  import numpy as np
  
  start1 = time.perf_counter()  # High-resolution timer
  start2 = time.process_time()  # Measures CPU time (ignores sleep/wait)
  
  X, y = make_classification(
      n_samples=10000,       # Number of examples
      n_features=20,        # Total features
      n_informative=5,      # Meaningful features
      n_redundant=2,        # Linearly dependent features
      n_classes=2,          # Binary classification
      n_clusters_per_class=2,  # Cluster count per class
      random_state=42
  )
  
  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  
  # Create parallelized Random Forest with 500 trees
  clf = RandomForestClassifier(
      n_estimators=1000,  # Number of trees
      n_jobs=-1,         # Use all available cores (-1 = all cores)
      verbose=1,         # Show progress
      random_state=42
  )
  
  # Train the model
  clf.fit(X_train, y_train)
  
  # Make predictions
  y_pred = clf.predict(X_test)
  
  # Evaluate accuracy
  print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
  
  end1 = time.perf_counter()
  end2 = time.process_time()
  
  print(f"Elapsed: {end1 - start1:.4f} seconds")  # Format to 4 decimal places
  print(f"CPU time: {end2 - start2:.4f} seconds")  # Format to 4 decimal places
  ```
  </details>

</details>

---

<details markdown="block">
<summary> Adaboost </summary>

## Adaboost

As discussed in [(Sagi and Rokach, 2017)](docs/Sagi_2018_Ensemble_learning_A_survey_Wire.pdf), the
main idea of AdaBoost is to focus on instances that were previously misclassified when training a new inducer. The
level of focus given is determined by a **weight** that is assigned to each instance in the training set. In the first iteration,
the same weight is assigned to all of the instances. In each iteration, the weights of misclassified instances are increased,
while the weights of correctly classified instances are decreased. In addition, weights are also assigned to the individual
base learners based on their overall predictive performance.

**AdaBoost** is a *dependent* ML method since each tree is an improvement over previous trees in the sequence. This is the opposite of *random forests* where the tree are grown independently.

<! ---See [What is adaboost: a friendly explanation](https://www.youtube.com/watch?v=AtYN8QP-U6w) for a clear and very simple example that shows how the weights is each weak learner are calculated. 

For clear details and nice illustrations, see https://medium.com/towards-data-science/adaboost-classifier-explained-a-visual-guide-with-code-examples-fc0f25326d7b--->

</details>

---

<details markdown="block">

<summary> Random Forests vs Adaboost </summary>

| **Feature**               | **Random Forest**                                                                 | **AdaBoost**                                                                 |
|---------------------------|-----------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| **Base Model Type**       | Full decision trees (variable depth)                                              | Decision stumps (1-node trees) or weak learners                              |
| **Training Approach**     | Parallel training via bagging (Bootstrap Aggregating)                             | Sequential training via boosting (weighted error correction)                 |
| **Data Sampling**         | Bootstrap samples with replacement for each tree                                  | Original data with instance reweighting based on errors                      |
| **Feature Selection**     | Random subset of features at each node                                            | Single feature per stump (weak learner focus)                                |
| **Model Weights**         | Equal voting weight for all trees                                                 | Weighted voting based on individual learner accuracy                         |
| **Overfitting Risk**      | Lower due to bagging and feature randomness                                       | Higher, especially with noisy data (focuses on error correction)             |
| **Complexity**            | High complexity per tree (full decision trees)                                    | Low complexity per stump (simple weak learners)                              |
| **Training Speed**        | Faster (parallelizable trees)                                                     | Slower (sequential dependency between learners)                              |
| **Noise Handling**        | Robust due to feature/tree diversity                                              | Sensitive (error correction amplifies noise impact)                          |
| **Key Strength**          | Generalization through diverse tree ensembles                                     | High accuracy through iterative error correction                             |
| **Best Use Case**         | Large datasets with mixed feature types                                           | Smaller datasets with clear patterns (low noise)                             |

</details>

---

<details markdown="block">
<summary> Gradient boosting </summary>

## Gradient boosting

Gradient Boost is also a *dependent* method, since each tree is an improvement of the earlier trees. Gradient Boost provides a framework to build an ensemble of trees based on an arbitrary loss function. In Gradient Boosting, each new tree is computed using a **simple classifier** (also called weak inducer, that just performs better than random) over the **residuals** from the previous model.

You can find a brief and informal description of the ideas behind gradient boosting at [Visual Guide to Gradient Boosted Trees (xgboost), 5'](https://www.youtube.com/watch?v=TyvYZ26alZs) (uses *MNIST data set*) and a nice and brief description of the maths of gradient boosting at [Gradient Boosting : Data Science's Silver Bullet (15')](https://www.youtube.com/watch?v=en2bmeB4QUo)

For details and very nice illustrations, look at the two following posts:

1. [Regression](https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-1-regression-2520a34a502)

2. [Classification](https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-2-classification-d3ed8f56541e)

</details>

---

<details markdown="block">

<summary> Gradient boosting vs Adaboost </summary>

| Feature | AdaBoost | Gradient Boosting |
| :-- | :-- | :-- |
| **Main Idea** | Focuses on misclassified samples by adjusting their weights after each iteration | Fits each new learner to the residual errors (negative gradients) of the previous model|
| **Loss Function** | Uses exponential loss (mainly for classification) | Can use any differentiable loss function (flexible for regression and classification) |
| **Weak Learners** | Typically uses shallow trees (decision stumps, depth=1) | Can use deeper trees (depth > 1) |
| **Weighting** | Assigns weights to both samples and learners; misclassified samples get higher weights, and stronger learners have more influence | All trees usually have equal weight; model update is additive|
| **Flexibility** | Less flexible (mainly for classification, some regression)| More flexible (supports various loss functions and tasks)|
| **Interpretability** | More intuitive; easy to understand the effect of reweighting | Less intuitive; based on gradient descent optimization |
| **Performance** | Fast and simple; can be sensitive to noisy data and outliers | Often achieves higher accuracy; better handles complex data but slower to train |
| **Adoption** | Legacy technique, less common in recent competitions. | Widely adopted, state-of-the-art in many ML tasks |

</details>

---

<details markdown="block">

<summary> The XGboost classifier, and an exercise </summary>

XGBoost introduces L1/L2 regularization to prevent overfitting, uses parallel processing for speed, handles missing values natively, and employs advanced tree pruning for better performance. In particular, XGBoost has built-in "sparsity-aware" techniques to automatically handle missing data, whereas standard gradient boosting requires preprocessing to handle missing values. XGBoost is engineered to handle large, complex datasets through cache-aware prefetching and out-of-core computing for data that doesn't fit in memory. 

Check the introductory video on [Complete Beginners Guide to XGBoost Models](https://www.youtube.com/watch?v=BJXt-WdeJJo). It uses the *iris data set* as an example.

Consider the Montesinho burned area data set described in https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T3_missing_data_categorical_scaling.md. The goal is to predict the variable `y` which has been discretized: y is 0 when the burned area is lower than 5 ha and it is 1 otherwise.

1. Adapt the  pipeline for preprocessing and classification available in the notebook https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/forest_fire.ipynb and replace the `RandomForestClassifier` by `XGBClassifier` which is available under the `xgboost` package;
2. Try varying some parameters of the XGBClassifier like `n_estimators`,  `learning_rate`, `max_depth` to understand how they affect the precision of the result and the computation time;
3. Still using the same pipeline, replace `train_test_split` for training and testing by cross validation with `StratifiedKFold` where stratification uses the response variable `y`

</details>

---

<details markdown="block">
<summary> Variable importance</summary>

## Interpretability and Variable importance

**Interpretability** is critically important in machine learning, acting as the bridge between high-performing algorithms and human trust, safety, and understanding. 
Since interpretability is a concept difficult to define precisely, people focus  on **variable importance**, a measure of the influence of each input feature to predict
the output. 

### Variable importance in random forests

Let's start with a particular family of models: trees and forests [Scornet, 2021](https://arxiv.org/pdf/2001.04295).  In Breiman (2001) original random forests, there exist two importance
measures:

1. **Mean Decrease Impurity**, MDI, or Gini importance, see Breiman (2002),
which sums up the gain associated to all splits performed along a given variable; and

2. **Mean Decrease Accuracy**, MDA, or **permutation importance**, see Breiman (2001), 
which shuffles entries of a specific variable in the test data set and computes the
difference between the error on the permuted test set and the original test set.

Because of its very definition, MDI is an importance measure that can be computed for trees
only, since it strongly relies on the tree structure, whereas MDA is an instantiation of
the permutation importance that can be used for any predictive model. Both measures
are used in practice even if they possess several major drawbacks.

- **MDI** is known to favor variables with many categories. Even when variables have the same number of categories,
MDI exhibits empirical bias towards variables that possess a category having a high frequency. MDI is also biased in presence of correlated features.

- **MDA** seems to exhibit less bias than MDI but tends to overestimate correlated features. 


**Exercise**: run and adapt scripts below to answer to the questions on the interpretation and comparision of MDI and MDA

1. Script to compute MDI for different classifiers for the Iris data set: https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/iris_mdi_importance_rf.py Note: this script allows you to easily remove features from the data set.

2. Script to compute permutation importance over the test data for the Iris data set:  https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/iris_permutation_importance_rf_ada_gb.py Note: this script allows you to easily remove features from the data set.

3. Compare the results for MDI and permutation importance:
- Create a scatter plot for the Iris data set so you can understand what is the correlation between variables for each class
- Compare MDI and permutation importance (MDA) for features which are highly correlated
- Try removing features with high importance and compute importance again to see the effect on the remaining features
- Conclude that importance is relative: one feature can be very important or not depending on the remaining features

### Variable importance in generic ML models: SHAP

There is  great interest in sound importance measures that can be applied to any ML model. **SHAP** (SHapley Additive exPlanations) is a game-theoretic approach to explain the output of any machine learning model by quantifying the contribution of each feature to individual predictions. It assigns an importance value (Shapley value) to each feature, representing how much it moves the model output from the average prediction. 

Check [Explainable AI explained! | #4 SHAP](https://www.youtube.com/watch?v=9haIOplEIGM) for a short and clear explanation on how SHAP works. The SHAP package provides many useful visualizations tools.


---

</details>
