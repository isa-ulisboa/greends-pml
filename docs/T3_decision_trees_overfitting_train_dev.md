# Tabular data

In the topics below, we consider that examples are of tabular type, where each example is described by a numeric vector. Formally, the $i$-th example is described by a vector  $(x_{i1}, \dots, x_{ik})$ of length $k$, for examples $i = 1, \dots, n$ and labels are $y_1, \dots,  y_n$  as before. Non-numerical attributes can be  converted into numerical ones so that assumption holds.

# Decision trees

This is a model for classification, where labels are categories. The goal, as for other ML models, is to be able to predict the label of a new example from its features (predictors). The first example uses the known Iris data set.

The example below shows a decision tree for the iris data set. The root node represents the 4-dimensional space defined by the variables sepal length,sepal width, petal length, petal width. This is a 3-class problem where labels are the varieties setosa, versicolor, virginica.

We call depth of the decision tree to the maximum number of splits to define a leaf node. Note that the code establishes `max_depth=4` to prevent the tree from growing more than 4 levels. The figure indicates the number of examples (or training samples) that lie in each node of the tree. [Script (link to Colab)](../notebooks/T3_basic_decision_tree.ipynb)

<!---
<details>
    
<summary> Click to expand code: Basic decision tree classifier, plot tree, sklearn </summary>

```python
from sklearn.datasets import load_iris
from sklearn import tree
from matplotlib import pyplot as plt
iris = load_iris()

X = iris.data
y = iris.target
print(' labels: ', iris.target_names)

#build decision tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4,min_samples_leaf=4)
#max_depth represents max level allowed in each tree, min_samples_leaf minumum samples storable in leaf node

#fit the tree to iris dataset
clf.fit(X,y)

#plot decision tree
fig, ax = plt.subplots(figsize=(10, 10)) #figsize value changes the size of plot
tree.plot_tree(clf,ax=ax,feature_names=['sepal length','sepal width','petal length','petal width'])
plt.show()
```    
</details>
--->

---

<details markdown="block">
<summary> Impurity and loss </summary>

## Impurity and loss

To build a tree, several questions arise:
1. Which feature should be tested at a node?
2. When should a node be declared a *leaf*?
4. If a leaf node is impure, how should the category label be assigned?
5. How should be missing data handled?
6. If the tree becomes 'too large' how can it be made smaller and simpler (pruning)?

To answer to those questions, we first need to define a measure of the quality of the model. As before, one defines a *loss* for a decision tree since the  ultimate goal is to find the model with the lowest loss.

There are two types of decision trees in ML:
1. **Classification trees**, where the labels are categorial as in the `iris`data set example. In that case, the **predicted** label is the most frequent label in the examples that lie the leaf node. For instance, if there are $[0,3,1]$ samples of varieties `[setosa, versicolor, virginica]` in a leaf node, then the label of the leaf node is `versicolor` and this is the predicted label $\hat{y}$ for all examples that lie in that region. To compute the *loss* one relies on the distribution of labels in the leaf node. For instance, the node with $[0,3,1]$ examples has estimated probabilities $\hat{p}_1=0$,  $\hat{p}_2=0.75$, $\hat{p}_3=0.25$ of been assigned to each one of the classes.
2. **Regression trees**, where the labels are continuous. In that case, the label for the node is the mean of all labels of examples that lie in that node, i.e. $\hat{y}=\bar{y}$. The *loss* for the *i*-th example is then the dissimilarity between $\bar{y}$ and $y_i$. For *regression trees* the usual *loss* functions  are `mse`and `mae`.

Let's see how the *loss* of a classification tree is computed in general and of a split in particular is computed. The loss is a measure of the impurity and the associated uncertainty at the leaf nodes of the tree. The higher the impurity of the leaf nodes, the greater the classification uncertainty.

For any given node of the tree, with $n_1,n_2,\dots,n_c$ examples of each class, the estimated probabilities for the $c$ different labels are:

$$\hat{p_1}=\frac{n_1}{n},\dots,\hat{p_c}=\frac{n_c}{n},$$

where $n$ is the total number of examples at the node. For classification trees, the `DecisionTreeClassifier` class in `scikit-learn` uses one of the following criteria:

1. `gini`: This is the default criterion and it measures the impurity of a set of samples as the probability of misclassifying a randomly chosen element from the set. The Gini *loss* for a node of the tree is given by  $G = 1 - \sum_{i=1}^c \hat{p_i}^2$, where $\hat{p_i}$ is the estimated probability of belonging to the *i*-th class. The minimum value that $G$ can take is 0 (no uncertainty) which only occurs if one of the estimated probabilities is 1 (pure node).
2. `entropy`: Entropy is a fundamental concept in machine learning that measures the level of uncertainty in a dataset. A higher entropy value indicates a more heterogeneous dataset with diverse classes, while a lower entropy signifies a more homogeneous and predictable  subset of data. It ranges from 0 to 1, with higher values indicating greater uncertainty. The entropy *loss* for a node of the tree is given by $E=-\sum_{i=1}^c \hat{p}_i \log_2 \hat{p}_i$. See for instance this [A Simple Explanation of Information Gain and Entropy](https://victorzhou.com/blog/information-gain/) with examples of binary decisions and calculations of entropy at each node.

Both measures range from 0 (minimum impurity, maximum certainty) to some maximum value (maximum impurity, minimum certainty). For instance, for a 2-class problem, maximum impurity is reached when $p_1=p_2=0.5$, where

$$G=1-0.5=0.5 {\rm ~and ~~} E= - 2 \times (0.5 \, \log_2 0.5)= - 2 \times (-0.5)=1.$$

When the node is split into two new children nodes, the loss function is calculated separately for each subset resulting from the split, and the *total loss* is the weighted sum of the losses of the subsets, where the weights are the fractions of samples in each subset. The split with the lowest total loss (i.e., the greatest reduction in entropy) is chosen as the best split. The expression for the loss of a split is the following, where $L$ can be either the entropy $E$ or the Gini criterion $G$.

$$L = \frac{n_{\rm left}}{n} \times L_{\rm left} + \frac{n_{\rm right}}{n} \times L_{\rm right} ~~~~~~~~~(1)$$

The rules above allow us to compute the loss for any tree computed from the data set. For each new split, Equation 1 allows us to update the *loss* of the whole tree.

For the two loss function above (`entropy` and `gini`), it is guaranteed that the *total loss* of the tree cannot increase for any possible split. Therefore, there is always a reduction (strict or not) in *loss*  resulting from a split. This is called *information gain*.

</details>

---

<details markdown="block">
<summary> Choosing the possible splits</summary>

## Choosing the possible splits

For continuous explanatory variables, all $n$ examples are ordered for the  *j*-th feature:

$$x_{j(1)} \le x_{j(2)} \dots \le  x_{j(n)}.$$

Hence, it is not necessary to test more than $n$ splits for each feature $j$. The spliting algorithm is just something like below.

---
Initialize $L$ as an empty list

For $j=1,\dots,k$

$~~~~$ For $i=1, \dots n$,

$~~~~$ $~~~~$ Consider the split $x_j \le x_{j(i)}$, compute its loss decrease and append it to $L$.

The best split is the split $x_j \le x_{j(i)}$ which has the lowest value in $L$.

---

For categorical explanatory variables, when there is no order along values, in principle all $2^m$ combinations of the $m$ distinct values that the variable can take should be considered as possible splits.

</details>

---

<details markdown="block">
<summary> Overfitting and regularization</summary>


## Overfitting and regularization

Decision trees are prone to *overfitting* since that if they grow enough they can approximate any decison rule with arbitrary precision. Therefore, there are different techniques to prevent decision trees of being  too large.

1. Criteria to stop growing the tree, i.e. a node should is not splited and becomes a leaf node. These are the most common hyper-parameters (check all hyper-parameters for [`sklearn.tree.DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)):
  - Maximum depth of the tree (e.g. 4);
  - Minimum leaf size, i.e., minimum number of examples that lie in a leaf (e.g. 3);
  - Maximum number of nodes (e.g. 20).

2. Automating Hyperparameter Selection
   
   The sklearn package provides robust tools for dividing the data and searching through hyperparameter combinations to find the configuration that generalizes best to unseen data:
   - train_test_split: Used to partition the original data set into training and development (validation) sets. The model learns on the training set, while the development set is used to evaluate performance across different hyperparameter settings.
    - GridSearchCV: This performs an "exhaustive search" over a specified parameter grid. For example, if you want to test max_depth values of [3, 5, 10] and min_samples_leaf values of [1, 5, 10], it will train and evaluate all 9 possible combinations.
    - RandomizedSearchCV: Instead of trying every possible combination, this samples a fixed number of parameter settings from specified distributions. This is often more efficient when dealing with a large number of hyperparameters.
    - Cross-Validation (cv): Usually integrated into the search functions above, this splits the training data into multiple "folds" to ensure the hyperparameter performance is consistent and not just a result of a lucky data split.

    A typical implementation looks like this:
    <details>
    
    <summary> Click to expand Python script</summary>
    
    ```python
    
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.tree import DecisionTreeClassifier

    # 1. Split data to save and independent subset of data for evaluation precision
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 2. Define the parameter grid
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_leaf': [1, 2, 5],
        'criterion': ['gini', 'entropy']
    }
    
    # 3. Initialize the search
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # 4. Access the best parameters
    print(f"Best parameters: {grid_search.best_params_}")

    # 5. Train the model with best hyper-parameters over all train+validation data

    # 6. Evaluate the final model over the independent test data set.
    ```

    </details>

4. Pruning. This is a regularization technique that consists on pruning the full grown tree to reduce its size. Pruning can be achieved by:
  - Adding a regularization hyper-parameter to the loss function, like $\alpha(|T|)$ where $\alpha$ is a function of the size (number of leaves) of the tree $T$. If one uses $L_\alpha=L+\alpha$ (consider that $\alpha >0$) instead of $L$ to determine the *loss*, then spliting a node might possibly cause an increase of $L_\alpha$.  If two leaves are pure and have the same label, aggregating them will lower $L_\alpha$ for $\alpha>0$. Pruning aggregates leaf nodes if that reduces $L_\alpha$.
  - Predicting a validation data set with the decision tree. Pruning consists of aggregating leaf nodes if that aggregation increases validation accuracy.

    See [Post pruning decision trees with cost complexity pruning of Scikit Learn](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html) for a detailed analysis on how to estimate the optimal regularization parameter $\alpha$ using train and developments data sets. 

</details>

<!---
<details markdown="block">
<summary> Choosing the best hyper-parameter values for a decision tree with a bias and variance plot</summary>

## Choosing the best hyper-parameter values for a decision tree with a bias and variance plot

When fitting a decision tree, one practical problem that arises is choosing the best values for the hyper-paramenters like `max_depth`. Using solely the training set does not make sense since this would lead to over-fitting. Therefore, this choice should rely on a development data set. To make the most informed decision one should take into account not just the average precision of the decision tree over the development data set but also its variance. Plotting the *bias* and *variance* of the model for a range of values of the hyper-parameter is a useful technique to decide on the best hyper-parameter value to use. 

**Bias** measures how much the predictions of a model differ from the true values. **Variance** measures how much the predictions of a model differ from each other. One possible technique to estimate  bias and variance is **cross-validation**.

In general, cross-validation is a technique used to evaluate the performance of a machine learning model. It involves splitting the dataset into multiple subsets, training the model on some of them, and testing it on the remaining subsets. This allows us to estimate the performance of the model on new, unseen data.

[**Script**](../notebooks/decision_tree_validation_curve.ipynb) relies on `validation_curve` from `scikit-learn` to estimate bias and variance of a classification model. In fact, the code applies a family of models (decision trees) that depend on the hyper-parameter `max_depth`.  Note that the synthetic data set is generated by `make_classification`. The output plot shows clearly the issua of **overfitting** since the estimated *train* accuracy keeps growing with the depth of the tree. However, the *validation* accuracy stabilizes for when the hyper-parameter `max_depth` is larger than 6. For this example, *bias* and *variance* are estimated from the validation scores. For instance, for the model with `max_depth=4`, the estimated bias is $\approx 0.013$, which corresponds to an estimated 87% accuracy, while the estimated variance is $\approx 0.0004$, which is the plotted standard deviation ($\approx 0.02$) squared.

Suggestion: try the code above using the `gini` instead of the `entropy` criterion for spliting.

</details>



