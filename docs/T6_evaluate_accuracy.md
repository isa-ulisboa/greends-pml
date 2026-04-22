
# Evaluating model performance

This is one of the most important topics im ML. Ourdays, it's pretty easy to build a prediction model but the model might be useless if it cannot be applied successfully to new examples.

<details markdown="block">
<summary>Performance evaluation metrics for classification </summary>

## Performance evaluation metrics for classification

### Confusion matrix

The confusion matrix, also called error matrix, is a very useful tool to evaluate the precision of a classifier.

To compute the error matrix for a classifier ${\bf f_{\bf w}}({\bf x})$ trained with a given training set of examples, the steps are the following.

1. Consider a test set of examples $({\bf x}, y)$ that were not used for training;

2. Predict the labels $\hat{y}={\bf f_{\bf w}}({\bf x})$ for all examples in the test set;

3. Compare the predicted labels $\hat{y}$ with the true labels $y$ and create a two-way table where the rows represent the actual labels $y$  and the columns represent the predicted labels $\hat{y}$.

The following code illustrated how to compute a confusion matrix for a classification task with two classes, labeled 0 and 1, and plot the result with `matplotlib`. The matrix compares the true labels of the examples `y_true` with the labels predicted by the classifier `y_pred`:

<details markdown="block">
<summary>Script to compute confusion matrix given actual and predicted values</summary>

  ```
  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  import numpy as np
  import matplotlib.pyplot as plt
  
  # Data
  y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
  y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
  
  # Plot confusion matrix
  ConfusionMatrixDisplay.from_predictions(
      y_true, 
      y_pred, 
      display_labels=['Zero', 'One'],
      cmap='Blues',
      colorbar=True
  )
  plt.title('Confusion Matrix')  # Optional title
  plt.tight_layout()
  plt.show()
  ```

</details>

### Accuracy metrics derived from the confusion matrix

In general, if there are $n$ different label values, the confusion matrix is $n \times n$. For simplicity, let's just consider the $2 \times 2$ error matrix, where correct predictions are called TP or TN, and the errors FP or FN.

|           | Predicted Positive | Predicted Negative |
|-----------|--------------------|--------------------|
| Actual Positive | TP=True Positive     | FN=False Negative    |
| Actual Negative | FP=False Positive| TN=True Negative|

<details markdown="block">
<summary>Metrics are computed from the confusion matrix</summary>

### accuracy, precision, recall, specificity, F1-score for a two-class problem

1. Classification **accuracy**.

$${\rm accuracy}=\frac{{\rm TP}+{\rm TN}}{{\rm TP}+{\rm FN}+{\rm FP}+{\rm TN}}.$$

If the number of actual positive examples (TP+FN) is very different from the number of negative examples (FP+TN), the largest number is going to dominate the result. 

For instance, suppose we want to predict if a given area was burned or not when actually 5% of some area is burned. Consider a trivial classifier that just labels all pixels as non-burned. Then that classifier would have a classification accuracy of 95%, which doesn't make much sense. For that example, the error matrix will look something that the following one if the total number of pixels is 10000.


|           | Predicted Burned | Predicted Non burned |
|-----------|--------------------|--------------------|
| Actual Burned | TP=0   | FN=500   |
| Actual Non burned | FP=0| TN=9500|

---

2. **Precision**, focused on predicted positives

$${\rm precision}=\frac{{\rm TP}}{{\rm TP}+{\rm FP}}.$$

This metric focusses only on the positive examples. Consider this other example, where one aims af finding greenhouses in a certain region (where 1\% of the total area is occupoid by greenhouses).


|           | Predicted Greenhouse | Predicted Other |
|-----------|--------------------|--------------------|
| Actual Greenhouse | TP=80   | FN=20    |
| Actual Other  | FP=10| TN=9890|

In that case, precision is just $80/(80+10) \approx 89\%$, while the overall classification accuracy is $99.7\%$.

Precision is the complement of **commission error**:

$${\rm CE}=\frac{{\rm FP}}{{\rm TP}+{\rm FP}}.$$

---

3. **Recall**, focused on actual positives, and also called **sensitivity** or **true positive rate (TPR)**

$${\rm recall}={\rm TPR}=\frac{{\rm TP}}{{\rm TP}+{\rm FN}}.$$

The denominator here is the total number of actual positives. This is an interesting metric if we are focused on having a very low error on missing an actual positive (a typical example is missing a tumor in medecine).

For the burned area example, the trivial classifier labels all pixels as non-burned has the worst possible outcome since it misses all actual positives, and therefore ${\rm recall}=0\%$. For the greenhouse example, we have ${\rm recall}=80\%$.

Recall is the complement of **omission error**:

$${\rm OE}=\frac{{\rm FN}}{{\rm TP}+{\rm FN}}.$$

For instance, one wants the *sensitivity* of a disease test to be high to ensure that sick people are detected.


---

4. **Specificity**, is focused on actual negatives, and is also called **true negative rate (TNR)**

$${\rm specificity}=\frac{{\rm TN}}{{\rm FP}+{\rm TN}}.$$

For instance, one wants the *specificity* of a disease test to be high to prevent healthy people from being labeled as sick.

A related metric is the **false positive rate (FPR)**, where the relation is ${\rm TNR} = 1 - {\rm FPR}$. 

$${\rm FPR}=\frac{{\rm FP}}{{\rm FP}+{\rm TN}}.$$

---

5. **F1 score**, which averages equally *precision* and *recall*

$${\rm F1~score}= 2 \times \frac{{\rm precision} \times {\rm recall}}{{\rm precision} + {\rm recall}}=\frac{{\rm 2\\, TP}}{{\rm 2\\, TP}+{\rm FP}+{\rm FN}}.$$

This is also known as the **Dice coefficient**. For the burned area example ${\rm F1~score}=0$ since in fact the F1 score is the *harmonic mean* of precision and recall. This metric still does not take into consideration true negatives (TN) and it is questionable since it gives the same importance to precision and recall.

6. **Mattews correlation coefficient (MCC)**

The Matthews correlation coefficient (MCC) is a reliable metric for evaluating binary (and multiclass) classifications in machine learning, offering a balanced measure even with imbalanced datasets. Ranging from -1 to +1, it considers all four confusion matrix entries (TP, TN, FP, FN). An MCC of +1 indicates perfect prediction, 0 is no better than random, and -1 indicates total disagreement. Unlike the F1-score, it is invariant to class swapping, which means it gives the same weight to any of the classes.

---

</details>

<details markdown="block">
<summary>Classification report</summary>


The precision metrics in a two-class classification problem depend on our decision about the *positive* class, which is typically the class of greater interest, and the *negative* class. For instance, if the problem is to determine burned areas over satellite imagery, the positive class would be *burned* and the nagative class would be *not burned*. Package `sklearn` offers a function that outputs a *classification report* that includes precision, recall and F1 score, for both possible labelings of the examples, i.e. considering both choices for the *positive* class.

The next **script** illustrates the use of `classification_report`:

  ```
  from sklearn.metrics import classification_report
  import numpy as np
  # Actual labels
  y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0])
  # Predicted labels
  y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
  # Compute confusion matrix
  report = classification_report(y_true, y_pred)
  print(report)
  ```
</details>

</details>

<details markdown="block">
<summary>ROC curve and AUC for two-class classification problems</summary>

## ROC curve and AUC for two-class classification problems

Consider the burned area problem, where the goal is to predict for each pixel if the class is *burned* or *not burned*. Suppose that we use a logistic regression as a classifier which returns a likelihood of being burn, between 0 and 1, for each pixel. Then, a threshold is applied to make the final decision. The usual threshold for prediction is 50% but there is no *a priori* reason to choose that threshold. 

Therefore, we can test our results for a range of thresholds. For instance, if the threshold is set as 70%, the model predicts an observation as positive only if the predicted probability is greater than 70%. Adjusting the threshold value changes some of the predicted labels and the overall performance of the classifier. Usually, a high threshold makes the prediction of the positive class less likely. This tends to increase both the false positive rate (FPR) and the true positive rate (TPR).

ROC curves typically feature true positive rate (TPR) on the Y axis, and false positive rate (FPR) on the X axis. This means that the top left corner of the plot is the “ideal” point - a FPR of zero, and a TPR of one (https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html).

<details markdown="block">
<summary>Example of a ROC curve</summary>

<img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/Roccurves.png" width="400" >
<img src="https://miro.medium.com/v2/resize:fit:702/format:webp/0*pF07ZmzBqbvkvqJO.png" width="400" >

</details>

The **AUC** is the area under the ROC curve. Its maximum value is 1, when the classifier return a correct decision regardless of the threshold value. *AUC* does not depend on the classification threshold, since it integrates all thresholds. It is used mostly to compare models.

[**Script** for drawing a ROC curve and computing the AUC from the classifier for the Wine Quality data set](https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/wine_quality_RocCurveDisplay.ipynb)

</details>
