# greends-pml
Links and exercises for the course Practical Machine Learning, Green Data Science, 2o semester 2023/2024

---
Instructor: Manuel Campagnolo, ISA/ULisboa

The course will follow a flipped classroom model. Work outside class will be based on the [Practical Deep Learning course](https://course.fast.ai/) and other Machine Learning resources. During classes, the notebooks (Python code) will be run on Google Colab.

The main materials for the course are:

- [Overview notebook](ML_overview_with_examples.ipynb) 
- Fast.ai
  - Introductory video (by Jeremy Howard, about 1h35) [Lesson "0": Practical Deep Learning for Coders (2020)](https://www.youtube.com/watch?v=gGxe2mN3kAg). Optional:  instructions described in the last part of the video for installing fastai in ubuntu for a machine with NVIDIA gpu are available at [https://course20.fast.ai/start_aws](https://course20.fast.ai/start_aws)
  - The lessons (videos by Jeremy Howard) available at [Practical Deep Learning for Coders 2022](https://course.fast.ai/): (1) Getting started, (2) Deployment; (3) Neural net foundations; (4) Natural Language (NLP); (5) From-scratch model; (6) Random forests; (7) Collaborative filtering; (8) Convolutions (CNNs); (9) Data ethics. Summaries of the lessons are also available at [https://course.fast.ai/](https://course.fast.ai/)
  - The notebooks for the 2022 course lessons, available at [https://github.com/fastai/course22](https://github.com/fastai/course22-web/tree/master/Lessons): look for lesson#.qmd file that lists the resources for the corresponding lesson. 
  - The online book [Deep Learning for Coders with Fastai and PyTorch: AI Applications Without a PhD](https://course.fast.ai/Resources/book.html). The examples on the book are not always the examples in the lessons. 
- Other Machine Learning resources:
  - [MIT 6.S191: Introduction to Deep Learning (2023)](https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)
  - [Stanford Lecture Collection | Convolutional Neural Networks for Visual Recognition (2017)](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) and [Notes for the Stanford course on Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)
  - [Stanford Machine Learning Full Course led by Andrew Ng (2020)](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU). Led by Andrew Ng, this course provides a broad introduction to machine learning and statistical pattern recognition. Topics include: supervised learning (generative/discriminative learning, parametric/non-parametric learning, neural networks, support vector machines); unsupervised learning (clustering, dimensionality reduction, kernel methods); learning theory (bias/variance tradeoffs, practical advice); reinforcement learning and adaptive control.
  - [Broderick: Machine Learning, MIT 6.036 Fall 2020](https://www.youtube.com/watch?v=ZOiBe-nrmc4); [Full lecture information and slides](http://tamarabroderick.com/ml.html)
  
---

Materials for course sessions
  
  - **(1) Session 'Introduction'**
    * Run [this notebook](test_GPU.ipynb) to compare CPU with CPU+GPU (try it in Colab: first, go to menu 'runtime/change runtime type' in Colab and choose GPU as hardware accelarator)
    
  - **(2) Session 'Getting started'** (to do before February 24)
    - Watch video: Lesson 1 of [Practical Deep Learning for Coders 2022](https://course.fast.ai/) 
    - Notebooks adapted for Colab from [https://github.com/fastai/course22](https://github.com/fastai/course22):
      - [Lesson1_00_is_it_a_bird_creating_a_model_from_your_own_data.ipynb](Lesson1_00_is_it_a_bird_creating_a_model_from_your_own_data.ipynb), where one builds a classifier for images of birds and forests.
      - [Lesson1_02_saving_a_basic_fastai_model.ipynb](Lesson1_02_saving_a_basic_fastai_model.ipynb)
    - Notebook for Chapter 1 of the book [Deep Learning for Coders with Fastai and PyTorch](https://course.fast.ai/Resources/book.html)
      
  - **(3) Session 'Deployment'** (to do before March 3rd)
    - **Watch video**: Lesson 2 of [Practical Deep Learning for Coders 2022](https://course.fast.ai/) 
    - **Notebook**: Edited notebook for Chapter 2 of the book [Lesson2_edited_book_02_production.ipynb](Lesson2_edited_book_02_production.ipynb), where one builds and improve a classifier for bears.
    - **Deploy a fastai classifier on HuggingFace Spaces with Gradio** [https://www.tanishq.ai/blog/gradio_hf_spaces_tutorial](https://www.tanishq.ai/blog/gradio_hf_spaces_tutorial). 
    - To do that, you need to sign up first to [HuggingFace](https://huggingface.co/spaces)
    - You can also watch [this short video (8'53) that shows how I deployed the bear classifier example trained in Colab on HuggingFace Places using Gradio](https://www.youtube.com/watch?v=QkUyjwue3f4). Suggestion: in the colab notebook and `app.py`, comment (with #) or remove line of code `img = PILImage.create(img)` in the definition of `predict` since the output of `PILImage.create` misses some needed properties. 
    - The notebook [tests/read_model_launch_gradio_in_colab.ipynb](tests/read_model_launch_gradio_in_colab.ipynb) is similar to `app.py` and shows how you can test it first on Colab before you deploy it on HuggingFace Spaces.  
    
  - **(4) Session 'Neural net foundations'** (to do before March 10)
    - **Watch video**: Lesson 3 of [Practical Deep Learning for Coders 2022](https://course.fast.ai/) **at least from ~23' to ~49'**.
    - **Colab notebook** for that part of the video: **How does a network really work?** (video from ~23' to ~49') [Lesson3_edited_04-how-does-a-neural-net-really-work.ipynb](Lesson3_edited_04-how-does-a-neural-net-really-work.ipynb). Uses a simple example (single variable function) to describe what we call the *loss*, what is *stochastic gradient descent* and how it can be implemented in Pytorch. It also describes and discusses the *ReLu* function.
           
  - **(5) Session 'From-scratch model' (part 1)** (to do before March 17: notice that Lesson 4 is postponed to a later date, so we move on to Lesson 5)
    - **Watch video**: Lesson 5 of [Practical Deep Learning for Coders 2022](https://course.fast.ai/), **at least up to 58'25''** (submitting results to Kaggle)
    - **Colab notebook** for the video: [Lesson5_edited_for_colab_linear_model_and_neural_net_from_scratch.ipynb](Lesson5_edited_for_colab_linear_model_and_neural_net_from_scratch.ipynb): follows the video up to 1:15'30'', but you only need to follow up to the end of the section 'Submitting results to Kaggle' (video ~58'25'').
    - In case you have trouble choosing the right Titanic data set from Kaggle, use `train.csv` from [titanic.zip](titanic.zip). This zip file also includes a file with extension `.ts` which can be loaded with `torch.load` and contains the numeric data matrix *X*  and the 0/1 response variable *y*.
    - Start thinking about getting your own tabular data set for a classification problem. If you want, you can look at [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets) for data sets that you find interesting. 

  - **(6) Session 'From-scratch model' (part 2)** 
    - Overview of the course with notebook [ML_overview_with_examples.ipynb](ML_overview_with_examples.ipynb) up to the Perceptron model.
    - (to do before March 24) **Watch the remainder part of the video** 58'25''- 1:15'30'': Lesson 5 of [Practical Deep Learning for Coders 2022](https://course.fast.ai/)
    - (to do before March 24) **Do assignment** [assignments/assignment_due_march_24.ipynb](assignments/assignment_due_march_24.ipynb)
      - You will need to look at [ML_overview_with_examples.ipynb](ML_overview_with_examples.ipynb), mostly the last part (Perceptron and Feed-forward neural network)
      - You will use the last part of [Lesson5_edited_for_colab_linear_model_and_neural_net_from_scratch.ipynb](Lesson5_edited_for_colab_linear_model_and_neural_net_from_scratch.ipynb), about adapting the one layer model to the deep learning model.
    
  - **(7) Session convolutional neural networks** 
    - Revision on how to adapt Perceptron code (with mini-batch) in [ML_overview_with_examples.ipynb](ML_overview_with_examples.ipynb) to create code from scratch for a feed-forward multi-layer neural network. This is necessary to answer part 2 of [assignments/assignment_due_march_24.ipynb](assignments/assignment_due_march_24.ipynb).
    - (do before March 31) **Watch video** of Lesson 5 of [Stanford Lecture Collection | Convolutional Neural Networks for Visual Recognition (2017)](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) (aprox. 1:08'). Some contents of the video: 9'07 (1st deep CNN diagram); 16' (input+kernel CNN); 21':23' (layers 1+2, activation maps, depth); 32'40 (stride); 34'54 (zero-pad); 45' (#parameters); 46' (common settings); 55'-56' (pooling layer); 1:01' (fully connected layer); 1:06' (demo url); 1:07'30 (summary, trend).

  - **(8) Session convolutional neural networks (continued)**
    - Revision of major concepts about CNNs
    - Brief explanation of residual units for `resnet` models -- details can be found in video of Lesson 9 (47'--1h03') of [Stanford Lecture Collection | Convolutional Neural Networks for Visual Recognition (2017)](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
    - Brief description of Encoder/Decoder architectures for image segmentation problems; the U-Net model.
    - (to do before April 14):
      - **Watch video** about image segmentation  of Lesson 11 (from 8' up to 32'53) of [Stanford Lecture Collection | Convolutional Neural Networks for Visual Recognition (2017)](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv). Video includes what is semantic segmentation, downsampling; upsampling; transpose convolution.
      - (optional, but recommended) If you want to try to do an image segmentation yourself using a U-Net, run [Image_Segmentation_with_Unet.ipynb](Image_Segmentation_with_Unet.ipynb).

- **(9) Session convolutional neural networks (conclusion); tabular data; feature engineering; categorial embeddings**
  - U-Net for image segmentation: example and comments in [Image_Segmentation_with_Unet.ipynb](Image_Segmentation_with_Unet.ipynb).
  - Tabular data; feature engineering; categorial embeddings [fast_ai_titanic.ipynb](fast_ai_titanic.ipynb)
  - (to do before April 21):
    - **Watch overview video** for topics already studied in class on 11:33 - Why deep learning?; 14:48 - The perceptron; 20:06 - Perceptron example; 23:14 - From perceptrons to neural networks; 29:34 - Applying neural networks; 32:29 - Loss functions; 35:12 - Training and gradient descent; 40:25 - Backpropagation; 44:05- Setting the learning rate; 48:09 - Batched gradient descent; 51:25 - Regularization: dropout and early stopping [MIT 6.S191: Introduction to Deep Learning 2023 edition](https://www.youtube.com/watch?v=QDX-1M5Nj7s).

- **(10) Fundamentals of classification and regression trees; Ensembles of classifiers**
  - Example of how to build a decision tree with `scikit-learn` (iris data) [ML_overview_with_examples.ipynb](ML_overview_with_examples.ipynb)
  - Q6
  - (to do before April 28):
    - **Watch  video** about [Lecture 10 - Decision Trees and Ensemble Methods | Stanford CS229: Machine Learning (Autumn 2018)](https://www.youtube.com/watch?v=wr9gUr-eWdA&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=10). You might want to skip 43'--47'50 on runtime and 1h15 to the end about boosting.

- **(11) Decision and regression trees; Ensembles of classifiers; Random Forests**
  - Overview on decision trees [ML_overview_with_examples.ipynb](ML_overview_with_examples.ipynb)
  - (to do before May 5):
    - **Watch  video** [MIT: Machine Learning 6.036, Lecture 12: Decision trees and random forests (Fall 2020) ](https://www.youtube.com/watch?v=ZOiBe-nrmc4): how to split the tree (24'50-37'20); Construct tree (up to 55'50); regularize and pruning (up to 1:07'); Ensemble methods (up to 1:20').
    - Submit project proposal.

- **(12) Assessing ML performance**
  - Q7
  - Confuson matrix, accuracy metrics, cross-validation, the ROC curve and AUC [ML_overview_with_examples.ipynb](ML_overview_with_examples.ipynb)
  - (to do before May 12):
    - **Watch  video** [Stanford CS229: Machine Learning (Autumn 2018), Lecture 8: Data Splits, Models & Cross-Validation](https://www.youtube.com/watch?v=rjbkWSTjHzM&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=8): Bias, variance, regularization (up to 19'), Other way of thinking of regularization; comparing algorithms  (37'45 to 1.02); small data sets; k-fold validation; leave-one-out (up to 1:17); feature selection (intro). You can skip from 19' to 37'45.
  
- **(13) Searching for the best model**
  - Q8 and correction of Q7
  - Techniques for searching a good model [A difficult classification problem: the Portuguese wine quality data set (extended)](select_best_model_wine_classification_extended.ipynb)
  - (to do before May 19):
    - **Watch  video** [Peter Prettenhofer - Gradient Boosted Regression Trees in scikit-learn](https://www.youtube.com/watch?v=IXZKgIsZRm0)

- **(14) Projects**
  - The class was used to advance on the class' projects.
  - (to do before May 26): Read the following sources. The idea is to focus on three ensemble methods: random forests, Adaboost and Gradient boosting (and its evolution XGBoost), and understand the main similarities and differences between them. You don't need to go into details and you can skip extra contents.
    - Read review article [Ensemble learning: a survey (2018)](docs/Sagi_2018_Ensemble_learning_A_survey_Wire.pdf)
    - Read [Ensemble methods with scikit-learn (scikit-learn documentation)](https://scikit-learn.org/stable/modules/ensemble.html#)

- **(15) Other ensemble methods; feature importance**
  - Ensemble models and feature importance: [A difficult classification problem: the Portuguese wine quality data set (extended)](select_best_model_wine_classification_extended.ipynb)
  
    
---
Some other useful links:
- [fastai documentation](https://docs.fast.ai/)
- [AIquizzes](https://aiquizzes.com/)
- [Harvard CS50 | Introduction to Programming with Python free course](https://pll.harvard.edu/course/cs50s-introduction-programming-python)
- [Walk with Fastai free version tutorial](https://walkwithfastai.com/)
- [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)

---
Additional (optional) notebooks:
- Comparison of image models (video from ~14' ): [Lesson3_which_image_models_are_best.ipynb](Lesson3_which_image_models_are_best.ipynb)
- Edited notebook for Chapter 4 of the book [Lesson3_edited_book_04_mnist_basics.ipynb](Lesson3_edited_book_04_mnist_basics.ipynb). 
        - The first part of the notebook uses the MNIST data set (MNIST contains images of handwritten digits) and provides an introduction to *tensors* and to *PyTorch* in particular. This introduction includes a discussion about shapes and dimensions, loss (dissimilarity), broadcasting, etc. That first part of the notebook *is not* discused in the video. 
        - The second part of the notebook is about *Stochastic Gradient Descent* with some simple examples of one variable functions. 
