# greends-pml
Links and exercises for the course Practical Machine Learning, Green Data Science, 2o semester 2023/2024

---
Instructor: Manuel Campagnolo, ISA/ULisboa

The course will follow a flipped classroom model. Work outside class will be based on the [Practical Deep Learning course](https://course.fast.ai/) and other Machine Learning resources. During classes, the notebooks (Python code) will be run on Google Colab.

Links for other resources:
  - [Fenix webpage](https://fenix.isa.ulisboa.pt/courses/aaap-283463546570956)
  - [Moodle ULisboa](https://elearning.ulisboa.pt/). The course is called *Practical Machine Learning*, with this [link](https://elearning.ulisboa.pt/course/view.php?id=8991). Students need to self-register in the course.
  - [Kaggle](https://www.kaggle.com/)

The main materials for the course are:

- [Overview notebook](https://github.com/isa-ulisboa/greends-pml/ML_overview_with_examples.ipynb) 
- Fast.ai
  - The lessons (videos by Jeremy Howard) available at [Practical Deep Learning for Coders 2022](https://course.fast.ai/): (1) Getting started, (2) Deployment; (3) Neural net foundations; (4) Natural Language (NLP); (5) From-scratch model; (6) Random forests; (7) Collaborative filtering; (8) Convolutions (CNNs). Summaries of the lessons are also available on that website.
  - The notebooks for the 2022 course lessons, available at [https://github.com/fastai/course22](https://github.com/fastai/course22-web/tree/master/Lessons): look for lesson#.qmd file that lists the resources for the corresponding lesson. 
  - The online book [Deep Learning for Coders with Fastai and PyTorch: AI Applications Without a PhD](https://course.fast.ai/Resources/book.html). The examples on the book are not always the examples in the lessons. 
- Other Machine Learning resources:
  - [MIT 6.S191: Introduction to Deep Learning (2023)](https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)
  - [Stanford Lecture Collection  Convolutional Neural Networks for Visual Recognition (2017)](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) and [Notes for the Stanford course on Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)
  - [Stanford Machine Learning Full Course led by Andrew Ng (2020)](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU). Led by Andrew Ng, this course provides a broad introduction to machine learning and statistical pattern recognition. Topics include: supervised learning (generative/discriminative learning, parametric/non-parametric learning, neural networks, support vector machines); unsupervised learning (clustering, dimensionality reduction, kernel methods); learning theory (bias/variance tradeoffs, practical advice); reinforcement learning and adaptive control.
  - [Broderick: Machine Learning, MIT 6.036 Fall 2020](https://www.youtube.com/watch?v=ZOiBe-nrmc4); [Full lecture information and slides](http://tamarabroderick.com/ml.html)
  
---

Sessions:
  
  - **Introduction** (Feb 23, 2024)
    - Examples of input data for machine learning problems: tabular data, images, text. See *Iris data set* example with the notebook [iris_regression_classification.ipynb](iris_regression_classification.ipynb)
    - Using Colab to run notebooks
    - Notebook adapted for Colab from [https://github.com/fastai/course22](https://github.com/fastai/course22):
      - [Lesson1_00_is_it_a_bird_creating_a_model_from_your_own_data.ipynb](Lesson1_00_is_it_a_bird_creating_a_model_from_your_own_data.ipynb), where one builds a classifier for images of birds and forests.
    - Assigments:
      - Create notebook on Colab to download images (to a Google Drive folder) with some prompt (e.g. 'corn leaf'), using a library other than `fastai` (e.g. some library that relies on DuckDuckGo or some other search engine). Each student should create a video (2' maximum) describing their code and showing that it runs on Colab, and submit the video until next Wednesday, Feb 28.
      - Watch video: Lesson 1 of [Practical Deep Learning for Coders 2022](https://course.fast.ai/) 
  
  - **Basic concepts** (Mar 1, 2024)
    - Discussion of the proposed solutions for the assignment of the previous class
    - Basic concepts in Machine learning: model and *loss*, *gradient descent*, for a simple regression problem. See [Overview notebook](ML_overview_with_examples.ipynb) and the code for a [simple example with a quadratic function](Lesson3_edited_04-how-does-a-neural-net-really-work.ipynb) adapted from the (Fastai 2022 course)[[https://github.com/fastai/course22](https://github.com/fastai/course22-web/tree/master/Lessons)].
    - Assignment:
      - Watch video: [MIT Introduction to Deep Learning 6.S191, 2023 edition](https://www.youtube.com/watch?v=QDX-1M5Nj7s&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=1&t=361s). There will be a questionnaire about some basic concepts discussed in the video. Contents: 11:33​ Why deep learning?; 14:48​ The perceptron; 20:06​ Perceptron example; 23:14​ From perceptrons to neural networks; 29:34​ Applying neural networks;  32:29​ Loss functions;  35:12​ Training and gradient descent; 40:25​ Backpropagation; 44:05​ Setting the learning rate; 48:09​ Batched gradient descent; 51:25​ Regularization: dropout and early stopping; 57:16​ Summary
    - Suggestion: Adapt the code in [simple example with a quadratic function](Lesson3_edited_04-how-does-a-neural-net-really-work.ipynb) to train a linear regression model $y=ax+b$ with just two parameters (instead of the three parameters of the quadratic function in the example). Compare the $a,b$ values that are obtained by
      - gradient descent after $N$ epochs considering the *MSE* (mean square error) loss function (instead of the *MAE* function in the example), with
      - the optimal ordinary least square linear regression coefficients that you can obtain for instance by fitting a `LinearRegression` with `scikit-learn`.

   -  **Linear regression vs perceptron; train, test and validation sets** (Mar 15, 2024)
      - The perceptron and the activation function; forward propagation
    
---
Some other useful links:
- [fastai documentation](https://docs.fast.ai/)
- [AIquizzes](https://aiquizzes.com/)
- [Harvard CS50 : Introduction to Programming with Python free course](https://pll.harvard.edu/course/cs50s-introduction-programming-python)
- [Walk with Fastai free version tutorial](https://walkwithfastai.com/)
- [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)

---
Additional (optional) notebooks:
- Comparison of image models (video from ~14' ): [Lesson3_which_image_models_are_best.ipynb](Lesson3_which_image_models_are_best.ipynb)
- Edited notebook for Chapter 4 of the book [Lesson3_edited_book_04_mnist_basics.ipynb](Lesson3_edited_book_04_mnist_basics.ipynb). 
        - The first part of the notebook uses the MNIST data set (MNIST contains images of handwritten digits) and provides an introduction to *tensors* and to *PyTorch* in particular. This introduction includes a discussion about shapes and dimensions, loss (dissimilarity), broadcasting, etc. That first part of the notebook *is not* discused in the video. 
        - The second part of the notebook is about *Stochastic Gradient Descent* with some simple examples of one variable functions. 
