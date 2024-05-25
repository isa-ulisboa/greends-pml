# greends-pml
Links and exercises for the course Practical Machine Learning, Green Data Science, 2o semester 2023/2024

---
Instructor: Manuel Campagnolo, ISA/ULisboa

The course will follow a flipped classroom model. Work outside class will be based on a range of Machine Learning resources including the book *Sebastian Raschka, Yuxi (Hayden) Liu, and Vahid Mirjalili. Machine Learning with PyTorch and Scikit-Learn. Packt Publishing, 2022*. During classes, the notebooks (Python code) will be run on Google Colab.

Links for class resources:
  - [Fenix webpage](https://fenix.isa.ulisboa.pt/courses/aaap-283463546570956). Academic page, where final results will be posted.
  - [Moodle ULisboa](https://elearning.ulisboa.pt/). Evaluation: assignments and questionnaires. The course is called *Practical Machine Learning*, with this [link](https://elearning.ulisboa.pt/course/view.php?id=8991). Students need to self-register in the course.
  - [Kaggle](https://www.kaggle.com/). Access to data; candidate problems for the final project.

[Overview notebook](https://github.com/isa-ulisboa/greends-pml/blob/main/ML_overview_with_examples.ipynb) This notebook provides an overview of the full course and contains pointers for other sources of relevant information and Python scripts.

**Sessions:** Each description below includes the summary of the topics covered in the session, as well as the description of assignments and links to videos or other materials that students should work through.

<details markdown="block">
<summary> Introduction (Feb 23, 2024) </summary>

The goal of the first class is to give an example of a complex machine learning problem that can easily be solved using freely available resources. The example uses the high level machine learning package `fastai`.

- Examples of input data for machine learning problems: tabular data, images, text. See *Iris data set* example with the notebook [iris_regression_classification.ipynb](iris_regression_classification.ipynb)
- Using Colab to run notebooks
- Notebook adapted for Colab from [https://github.com/fastai/course22](https://github.com/fastai/course22):
  - [Lesson1_00_is_it_a_bird_creating_a_model_from_your_own_data.ipynb](Lesson1_00_is_it_a_bird_creating_a_model_from_your_own_data.ipynb), where one builds a classifier for images of birds and forests.
- Assigments:
  - **Assignment #1**: Create notebook on Colab to download images (to a Google Drive folder) with some prompt (e.g. 'corn leaf'), using a library other than `fastai` (e.g. some library that relies on DuckDuckGo or some other search engine). Each student should create a video (2' maximum) describing their code and showing that it runs on Colab, and submit the video until next Wednesday, Feb 28.
  - Watch video: Lesson 1 of [Practical Deep Learning for Coders 2022](https://course.fast.ai/) 

</details>

<details markdown="block">
<summary> Basic concepts (Mar 1, 2024): model, loss, gradient descent </summary>

The goal of the following classes up to April 12 is to understand how deep learning models can be trained and used to solve regression and classification problems. We start by applying the machine learning approach to well-known statistical problems like linear regression to illustrate the stepwise approach followed in ML. We use synthetic data generated from a linear or quadratic regression, where one can control the underlying model and the amout of noise. Then, we consider the  `Iris` tabular data set with 4 explanatory variables and categorical label that can be one of three species.

- Discussion of the proposed solutions for the assignment of the previous class
- Basic concepts in Machine learning: model and *loss*, *gradient descent*, for a simple regression problem. See [Overview notebook](ML_overview_with_examples.ipynb) and see the code for a simple example with a quadratic function in notebook [Lesson3_edited_04-how-does-a-neural-net-really-work.ipynb](Lesson3_edited_04-how-does-a-neural-net-really-work.ipynb). This note book is adapted from the (Fastai 2022 course) [https://github.com/fastai/course22](https://github.com/fastai/course22-web/tree/master/Lessons).
- Assignment:
  - Watch video: [MIT Introduction to Deep Learning 6.S191, 2023 edition](https://www.youtube.com/watch?v=QDX-1M5Nj7s&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=1&t=361s). There will be a questionnaire (**Questionnaire #2**) about some basic concepts discussed in the video. Contents: 11:33​ Why deep learning?; 14:48​ The perceptron; 20:06​ Perceptron example; 23:14​ From perceptrons to neural networks; 29:34​ Applying neural networks;  32:29​ Loss functions;  35:12​ Training and gradient descent; 40:25​ Backpropagation; 44:05​ Setting the learning rate; 48:09​ Batched gradient descent; 51:25​ Regularization: dropout and early stopping; 57:16​ Summary
- Suggestion: Adapt the code in the simple example with a quadratic function in notebook [Lesson3_edited_04-how-does-a-neural-net-really-work.ipynb](Lesson3_edited_04-how-does-a-neural-net-really-work.ipynb) to train a linear regression model $y=ax+b$ with just two parameters (instead of the three parameters of the quadratic function in the example). Compare the $a,b$ values that are obtained by
  - gradient descent after $N$ epochs considering the *MSE* (mean square error) loss function (instead of the *MAE* function in the example), with
  - the optimal ordinary least square linear regression coefficients that you can obtain for instance by fitting a `LinearRegression` with `scikit-learn`.

</details>

<details markdown="block">
<summary> Linear regression examples (Mar 15, 2024): epochs, perceptron, batches, train and test, overfitting</summary>

This session extends the previous class. We discuss some additional core ML concepts and we extend the approach to classification problems (discrete labels). The model (the *perceptron*) is still very simple and closely related to linear regression. 
  
- Discussion of the suggested assignment from last class
- Code for gradient descent using PyTorch; epochs and batches
- The perceptron and gradient descent: an `iris` data set example with animation
- Mini-batch example
- Train and test data; overfitting
- **Assignment #3**:
  1. adapt the `Perceptron` class in [Overview notebook](https://github.com/isa-ulisboa/greends-pml/blob/main/ML_overview_with_examples.ipynb) to address the two following goals. Describe the changes on the code and the effect of using mini-batches on the search of the best solution, as an alternative to stochastic gradient descent.
      - It uses mini-batches;
      - It computes and displays (on the animation, similarly to the iterations) the loss over a test set.
  2. Backup assignment (if the student is not able to do assignment #1 above).  Find a real data set adequate for a (simple or multiple) regression problem. Adapt the script based on `PyTorch` discussed in class an available in [Overview notebook](https://github.com/isa-ulisboa/greends-pml/blob/main/ML_overview_with_examples.ipynb) to solve the problem. Discuss the train and test losses plots along iterations and indicate what is  the best set of parameters (including number of epochs) that you found for the problem at hand.
  3. Each student should create a video (4' maximum for #1 and 3' maximum for 2#) and submit the video until next Thursday, Mar 21st, at noon.
</details>


<details markdown="block">
<summary> Regression vs classification problems; assessing ML performance (Mar 22, 2024): cross-entropy, confusion matrix, accuracy metrics</summary>

The main goal of this session is to understand how one can evaluate the accuracy of a classifier.
  
- Discussion of previous assignment from last class: example of gradient descent with mini batches for the `iris` data set;
- Perceptron and convergence; linearly separable sets
- Classification problems: scores and the *softmax* function; cross-entropy
- Assessing ML performance: confusion matrix, accuracy metrics
- Suggestion: watch the series of videos [https://www.3blue1brown.com/topics/neural-networks](https://www.3blue1brown.com/topics/neural-networks) which introduce neural networks in a pretty informal way, with very nice animations. The example that is used along the videos is from the `MNIST` database (Modified National Institute of Standards and Technology database), a large database of handwritten digits that is commonly used for training various image processing systems. Each example is a 28 by 28 image (784 pixels).
- **Assignment #4**:
  - Adapt the *Perceptron* code for the Iris data available in the [Overview notebook](https://github.com/isa-ulisboa/greends-pml/blob/main/ML_overview_with_examples.ipynb) to output the confusion matrix, using 20% of the data set for validation. Compute also the classification accuracy, precision, recall and F1-score for the solution.
  - Each student should create a video (3' maximum) and submit the video and the link to the GitHub repository and the file where the modified script is available.
  - Submission deadline: Thursday, March 28.
</details>

<details markdown="block">
<summary> Neural networks (April 5, 2024): an implementation with PyTorch</summary>

In this session we extend the *perceptron* to a more complex model with multiple layers, called a *neural network*. We discuss how a neural network can be created and trained with PyTorch. Two data sets are used to illustrate the construction: the tabular `Iris` data set that had been used before, and a more complex data set (`MNIST`) where examples are images, but at this point are read just as vectors of numbers in a similar tabular way to the `Iris` data set.
  
- Discussion of previous assignment;
- Neural networks with $n$ layers
- Model parameters: dropout, momentum, regularization, etc
- Examples with `Iris` and `MNIST` data sets.
- Suggestion: watch Data Umbrella [video](https://www.youtube.com/watch?v=B5GHmm3KN2A) with Sebastian Raschka, which is a friendly **Introduction to PyTorch tutorial**, that revisits many things that were discussed in class. Sebastian Raschka is a author of the books *Python Machine Learning 3rd Edition* and *Machine Learning with PyTorch and Scikit-Learn*. In this video, the NN model is a convolutional model to be applied to images, and therefore it has a feature extraction component (not yet discussed in class), followed by a multi layer perceptron component (already discussed in class). The additional feature extraction component that uses convolutions will be discussed in the next class, so the video is also a bit of an introduction to that.
- **Assignment #5**:
  - Adapt the *Script that implements a neural network with PyTorch (over the iris or mnist datasets)* code available on [Overview notebook](https://github.com/isa-ulisboa/greends-pml/blob/main/ML_overview_with_examples.ipynb) such that it is implemented with `TensorFlow` instead of `PyTorch`. Adjust the parameters to try to obtain a global accuracy close to 90% for the `MNIST` dataset. 
  - Each student should create a video (3' maximum) explaing which were the major changes that were made on the script and submit the video and the link to file in their GitHub repository where the modified script is available.
  - Submission deadline: Wednesday, April 10.
</details>

<details markdown="block">
<summary> Convolutional neural networks (April 12, 2024): parameters for convolutional layers ; an implementation with PyTorch</summary>

In this session, we improve on the model used in the previous session for the `MNIST` data set. Since the examples are images, it makes sense to explore the spatial context within each image. This can be done with convolutions over the images. Therefore, we add 2D-convolution and maxpool layers to the previous model and create a convolutional neural network using PyTorch.

- Improving the PyTorch code for NNs (making it more modular)
- Convolutional Neural networks
- Parameters: kernel, padding, stride, pooling
- Example of application for the `MNIST` data set.
- **Assignment #6**:
  - Adapt the *Script that implements a convolutional neural network with PyTorch over the mnist 8 by 8 practice data set* available on [Overview notebook](https://github.com/isa-ulisboa/greends-pml/blob/main/ML_overview_with_examples.ipynb) to classify images from the [CIFAR-10](https://www.kaggle.com/c/cifar-10/) data set which contains 60000 32 by 32 color images. You need to access/download the data and you need to adjust the model parameters for the input and hidden layers. The CIFAR-10 color images have 3 channels, H=32, and W=32. You are not supposed to use other pre-defined and pre-trained models as in many examples that you can find on-line. You're expected to adapt the model in the script. Getting a high accuracy is not a goal. The goal is to be able to adapt the code and explain in the video how you did it. Use a small number of epochs (start with only 5 perhaps) since training will take much longer than in the examples we have seen in class so far.
  - Each student should create a video (3' maximum) explaning which were the major changes that were made on the script and submit the video and the link to file in their GitHub repository where the modified script is available.
  - Submission deadline: Wednesday, April 24.
</details>

<details markdown="block">
<summary> Transfer learning (April 24, 2024): using and fine-tuning pre-trained models</summary>

We have seen how machine learning models are created and trained with PyTorch. However, when applying our model (e.g. a CNN) to a larger data set (e.f. CIFAR10) we encounter several problems like: 
1. the accuracy is low because the model is not good enough,
2. training from scratch requires a lot of computational resources.

Similarly to the first session (*Introduction*) where we discussed a short script using the high level package `fastai` to implement a pre-trained convolutional neural network and apply it to classify images downloaded from the internet, we will adapt the code we discussed earlier to read and improve a pre-trained model called `Resnet18`. Here, we will see how to access a pre-trained model in PyTorch, and fine-tuned it to our data set. This will address both concerns listed above.

- Transfer learning: watch [video bt Andrew Ng](https://www.youtube.com/watch?v=yofjFQddwHE); discuss with an example how to adapt and fine tune a `Resnet18` model to classify the CIFAR-10 data set.

- **Assignment #7** (identification of corn diseases):
  - Read carefully the notebook  [corn leaf disease](https://www.kaggle.com/code/emrearslan123/corn-leaf-disease-detection-with-resnet-pytorch) which classifies with high precision images of corn leaves into 4 classes: *Blight*, *Common_Rust*, *Gray_Leaf_Spot* and *Healthy*;
  - Implement that code or some similar code either on Colab or on your own machine and compare the your results with the results reported in the notebook for the same *corn diseases* data set; If you're not able to run it on GPU on Colab, you might try to run it on CPU on your own laptop/desktop for a few epochs;
  - You need to include an instruction in your code to save the model like `torch.save(model.state_dict(), "model.pth")` in the notebook above (or you can save it in a different format like the *pickle* format with extension `.pkl`), so the trained model can be re-used later;
  - Create a video (3' approx) explaning the main novelties in the code in comparison to what has been already discussed in class and possibly the difficulties you ran into. In particular, focus on the following aspects: reading and organizing data; pre-processing data before deep learning; original image size; adapting the Resnet18 to the problem at hand; moving data and model to the computing device; saving the model; discuss results and computational requirements. Submit the video the link to your script either in your GitHub repository or on Colab; report your estimated precision (you can just copy/paste the output of `classification_report(test.targets, all_preds)`).
  - Submission deadline: Wednesday, May 1st.

</details>

<details markdown="block">
<summary> Production (May 3, 2024): saving and deploying models with gradio</summary>


- Discussion of *assignment #7*. Check the proposed [Colab notebook for the problem](corn_leaf_disease_detection_with_resnet_pytorch.ipynb). In particular, it was discussed how to save the deep learning model so it can be deployed.
- Create your own Hugging Face space;
- Create an interactive app that runs locally and can also be deployed on your Hugging Face space that reads an image and returns the size of that image;
- How to create an app on Hugging Face spaces for the deployment of the classifier trained and saved in *assignment #7*. See code in [Overview notebook](https://github.com/isa-ulisboa/greends-pml/blob/main/ML_overview_with_examples.ipynb). Check also the links included in that notebook about deploying models with `Gradio`.
- **Assignment #8**: 
  - Each student should create a video (3' maximum) explaning how they saved their model, and how they used `gradio` and Hugging Face spaces for deployment. Submit the video and the link to your public Hugging Face space repository where files can be read, i.e. a link like (https://huggingface.co/spaces/mcampagnolo/test2024/tree/main). It is expected that you explain how you obtained the model file and the other files you had to upload to Hugging Face, and which files had to be created so your app would run. You are encouraged to make changes on your app to improve it. Submission deadline: Wednesday, May 8st.
</details>


<details markdown="block">
<summary> Tabular data (May 10, 2024): preprocess tabular data</summary>


- Discussion of  *assignment #7* and *assignment #8*.
- Brief discussion about (not free) platforms for machine learning (ML) like [Vertex AI](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform),  [MS Azure](https://azure.microsoft.com/en-us/products/machine-learning), [AWS Sagemaker](https://aws.amazon.com/sagemaker/), etc.
- Tabular data:
  - Pre-processing with `pandas`and `sklearn`: See Chap 4 notebook at [https://github.com/rasbt/machine-learning-book/](https://github.com/rasbt/machine-learning-book/) and an application to the *Wine* data set [https://archive.ics.uci.edu/dataset/109/wine](https://archive.ics.uci.edu/dataset/109/wine).
  -  *Wine quality* data set [https://archive.ics.uci.edu/dataset/186/wine+quality](https://archive.ics.uci.edu/dataset/186/wine+quality) with two different response variables: color (white or red) and quality (score between 0 and 10). Explore the data as in the *Wine*  example.
</details>

<details markdown="block">
<summary> Feature engineering and data visualization (May 17, 2024): t-SNE, UMAP, processing pipeline</summary>
  
- Final projet description
- Discuss *Script to apply dimensionality reduction techniques t-SNE, UMAP and LDA to several data sets*  in the [Overview notebook](https://github.com/isa-ulisboa/greends-pml/blob/main/ML_overview_with_examples.ipynb)
- Pre-processing and feature engineering with `sklearn`. See example [here](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html) with the  `titanic` data set.
- Processing pipelines: See Chap 6 notebook at [https://github.com/rasbt/machine-learning-book/](https://github.com/rasbt/machine-learning-book/)
- Brief description of the *decision tree* classifier.
  
</details>

<details markdown="block">
<summary> Random forests (May 23, 2024)</summary>
  
- Decision trees: see corresponding section in the [Overview notebook](https://github.com/isa-ulisboa/greends-pml/blob/main/ML_overview_with_examples.ipynb)
- Ensemble methods and random forests: see corresponding section in the [Overview notebook](https://github.com/isa-ulisboa/greends-pml/blob/main/ML_overview_with_examples.ipynb)
- **Assignment #9**:
  - Consider the *wine quality data set*, where the response variable is *quality* and both red and white wines are included in the data set;
  - The goal is to predict the quality from the explanatory variables;
  - You can preprocess the data and perform feature engineering;
  - The goal is to develop a classifier with overall accuracy over the test set larger than 55% for the problem using all the original quality classes;
  - You should exhibit and discuss the confusion matrix over the test set;
  - You should also show and discuss a plot that shows how the overal accuracy varies with some parameter like `max_depth`, `max_leaf_nodes` or another parameter designed to avoid overfitting
  - Create a video (3' approx.) describing your script and your results according and submit the video and the link to the file in your GitHub repository where the script is available (Submission deadline: Thursday, May 30th, 2pm)

  
</details>

---

<details markdown="block">
<summary> Main on-line resources </summary>

- PyTorch
  - Sebastian Raschka, Yuxi (Hayden) Liu, and Vahid Mirjalili. Machine Learning with PyTorch and Scikit-Learn. Packt Publishing, 2022. See the presentation [webpage](https://sebastianraschka.com/blog/2022/ml-pytorch-book.html) and [GitHub repository](https://github.com/rasbt/machine-learning-book)
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
