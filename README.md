# greends-pml
Links and exercises for the course Practical Machine Learning, Green Data Science, 2o semester 2023/2024

---
Instructor: Manuel Campagnolo, ISA/ULisboa

The course will follow a flipped classroom model. Work outside class will be based on the [Practical Deep Learning course](https://course.fast.ai/) and other Machine Learning resources. During classes, the notebooks (Python code) will be run on Google Colab.

Links for other resources:
  - [Fenix webpage](https://fenix.isa.ulisboa.pt/courses/aaap-283463546570956)
  - [Moodle ULisboa](https://elearning.ulisboa.pt/). The course is called *Practical Machine Learning*, with this [link](https://elearning.ulisboa.pt/course/view.php?id=8991). Students need to self-register in the course.
  - [Kaggle](https://www.kaggle.com/)

[Overview notebook](https://github.com/isa-ulisboa/greends-pml/blob/main/ML_overview_with_examples.ipynb) 

**Sessions:**

<details markdown="block">
<summary> Introduction (Feb 23, 2024) </summary>

- Examples of input data for machine learning problems: tabular data, images, text. See *Iris data set* example with the notebook [iris_regression_classification.ipynb](iris_regression_classification.ipynb)
- Using Colab to run notebooks
- Notebook adapted for Colab from [https://github.com/fastai/course22](https://github.com/fastai/course22):
  - [Lesson1_00_is_it_a_bird_creating_a_model_from_your_own_data.ipynb](Lesson1_00_is_it_a_bird_creating_a_model_from_your_own_data.ipynb), where one builds a classifier for images of birds and forests.
- Assigments:
  - Create notebook on Colab to download images (to a Google Drive folder) with some prompt (e.g. 'corn leaf'), using a library other than `fastai` (e.g. some library that relies on DuckDuckGo or some other search engine). Each student should create a video (2' maximum) describing their code and showing that it runs on Colab, and submit the video until next Wednesday, Feb 28.
  - Watch video: Lesson 1 of [Practical Deep Learning for Coders 2022](https://course.fast.ai/) 

</details>

<details markdown="block">
<summary> Basic concepts (Mar 1, 2024): model, loss, gradient descent </summary>

- Discussion of the proposed solutions for the assignment of the previous class
- Basic concepts in Machine learning: model and *loss*, *gradient descent*, for a simple regression problem. See [Overview notebook](ML_overview_with_examples.ipynb) and see the code for a simple example with a quadratic function in notebook [Lesson3_edited_04-how-does-a-neural-net-really-work.ipynb](Lesson3_edited_04-how-does-a-neural-net-really-work.ipynb). This note book is adapted from the (Fastai 2022 course) [https://github.com/fastai/course22](https://github.com/fastai/course22-web/tree/master/Lessons).
- Assignment:
  - Watch video: [MIT Introduction to Deep Learning 6.S191, 2023 edition](https://www.youtube.com/watch?v=QDX-1M5Nj7s&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=1&t=361s). There will be a questionnaire about some basic concepts discussed in the video. Contents: 11:33​ Why deep learning?; 14:48​ The perceptron; 20:06​ Perceptron example; 23:14​ From perceptrons to neural networks; 29:34​ Applying neural networks;  32:29​ Loss functions;  35:12​ Training and gradient descent; 40:25​ Backpropagation; 44:05​ Setting the learning rate; 48:09​ Batched gradient descent; 51:25​ Regularization: dropout and early stopping; 57:16​ Summary
- Suggestion: Adapt the code in the simple example with a quadratic function in notebook [Lesson3_edited_04-how-does-a-neural-net-really-work.ipynb](Lesson3_edited_04-how-does-a-neural-net-really-work.ipynb) to train a linear regression model $y=ax+b$ with just two parameters (instead of the three parameters of the quadratic function in the example). Compare the $a,b$ values that are obtained by
  - gradient descent after $N$ epochs considering the *MSE* (mean square error) loss function (instead of the *MAE* function in the example), with
  - the optimal ordinary least square linear regression coefficients that you can obtain for instance by fitting a `LinearRegression` with `scikit-learn`.

</details>

<details markdown="block">
<summary> Linear regression examples (Mar 15, 2024): epochs, perceptron, batches, train and test, overfitting</summary>
  
- Discussion of the suggested assignment from last class
- Code for gradient descent using PyTorch; epochs and batches
- The perceptron and gradient descent: an `iris` data set example with animation
- Mini-batch example
- Train and test data; overfitting
- Assignment:
  1. adapt the `Perceptron` class in [Overview notebook](https://github.com/isa-ulisboa/greends-pml/blob/main/ML_overview_with_examples.ipynb) to address the two following goals. Describe the changes on the code and the effect of using mini-batches on the search of the best solution, as an alternative to stochastic gradient descent.
      - It uses mini-batches;
      - It computes and displays (on the animation, similarly to the iterations) the loss over a test set.
  2. Backup assignment (if the student is not able to do assignment #1 above).  Find a real data set adequate for a (simple or multiple) regression problem. Adapt the script based on `PyTorch` discussed in class an available in [Overview notebook](https://github.com/isa-ulisboa/greends-pml/blob/main/ML_overview_with_examples.ipynb) to solve the problem. Discuss the train and test losses plots along iterations and indicate what is  the best set of parameters (including number of epochs) that you found for the problem at hand.
  3. Each student should create a video (4' maximum for #1 and 3' maximum for 2#) and submit the video until next Thursday, Mar 21st, at noon.
</details>


<details markdown="block">
<summary> Regression vs classification problems; assessing ML performance (Mar 22, 2024): cross-entropy, confusion matrix, accuracy metrics</summary>
  
- Discussion of previous assignment from last class: example of gradient descent with mini batches for the `iris` data set;
- Perceptron and convergence; linearly separable sets
- Classification problems: scores and the *softmax* function; cross-entropy
- Assessing ML performance: confusion matrix, accuracy metrics
- Assignment:
  - Adapt the *Perceptron* code for the Iris data available in the [Overview notebook](https://github.com/isa-ulisboa/greends-pml/blob/main/ML_overview_with_examples.ipynb) to output the confusion matrix, using 20% of the data set for validation. Compute also the classification accuracy, precision, recall and F1-score for the solution.
  - Each student should create a video (3' maximum) and submit the video and the link to the GitHub repository and the file where the modified script is available.
  - Submission deadline: Thursday, March 28.
</details>
---

<details markdown="block">
<summary> Main on-line resources </summary>

- Fast.ai
  - The lessons (videos by Jeremy Howard) available at [Practical Deep Learning for Coders 2022](https://course.fast.ai/): (1) Getting started, (2) Deployment; (3) Neural net foundations; (4) Natural Language (NLP); (5) From-scratch model; (6) Random forests; (7) Collaborative filtering; (8) Convolutions (CNNs). Summaries of the lessons are also available on that website.
  - The notebooks for the 2022 course lessons, available at [https://github.com/fastai/course22](https://github.com/fastai/course22-web/tree/master/Lessons): look for lesson#.qmd file that lists the resources for the corresponding lesson. 
  - The online book [Deep Learning for Coders with Fastai and PyTorch: AI Applications Without a PhD](https://course.fast.ai/Resources/book.html). The examples on the book are not always the examples in the lessons. 
- Other Machine Learning resources:
  - [MIT 6.S191: Introduction to Deep Learning (2023)](https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)
  - [Stanford Lecture Collection  Convolutional Neural Networks for Visual Recognition (2017)](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) and [Notes for the Stanford course on Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)
  - [Stanford Machine Learning Full Course led by Andrew Ng (2020)](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU). Led by Andrew Ng, this course provides a broad introduction to machine learning and statistical pattern recognition. Topics include: supervised learning (generative/discriminative learning, parametric/non-parametric learning, neural networks, support vector machines); unsupervised learning (clustering, dimensionality reduction, kernel methods); learning theory (bias/variance tradeoffs, practical advice); reinforcement learning and adaptive control.
  - [Broderick: Machine Learning, MIT 6.036 Fall 2020](https://www.youtube.com/watch?v=ZOiBe-nrmc4); [Full lecture information and slides](http://tamarabroderick.com/ml.html)

</details>
 
<details markdown="block">
<summary> Some other useful links </summary>

- [fastai documentation](https://docs.fast.ai/)
- [AIquizzes](https://aiquizzes.com/)
- [Harvard CS50 : Introduction to Programming with Python free course](https://pll.harvard.edu/course/cs50s-introduction-programming-python)
- [Walk with Fastai free version tutorial](https://walkwithfastai.com/)
- [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)

</details>
