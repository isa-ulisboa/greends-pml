# greends-pml
Links and exercises for the course Practical Machine Learning, Green Data Science, 2o semester 2022/2023

---
Instructor: Manuel Campagnolo, ISA/ULisboa

The course will follow a flipped classroom model, and most of the work will be based on the [Practical Deep Learning course](https://course.fast.ai/). During classes, the notebooks (Python code) will be run on Google Colab.

The main materials for the course are:

- Introductory video (by Jeremy Howard, about 1h35) [Lesson "0": Practical Deep Learning for Coders (2020)](https://www.youtube.com/watch?v=gGxe2mN3kAg). Optional:  instructions described in the last part of the video for installing fastai in ubuntu for a machine with NVIDIA gpu are available at [https://course20.fast.ai/start_aws](https://course20.fast.ai/start_aws)
- The lessons (videos by Jeremy Howard) available at [Practical Deep Learning for Coders 2022](https://course.fast.ai/):
  1. Getting started
  2. Deployment
  3. Neural net foundations
  4. Natural Language (NLP)
  5. From-scratch model
  6. Random forests
  7. Collaborative filtering
  8. Convolutions (CNNs)
  9. Data ethics
- Summaries of the lessons are also available at [https://course.fast.ai/](https://course.fast.ai/)
- The notebooks for the 2022 course lessons, available at [https://github.com/fastai/course22](https://github.com/fastai/course22-web/tree/master/Lessons): look for lesson#.qmd file that lists the resources for the corresponding lesson. 
- The online book [Deep Learning for Coders with Fastai and PyTorch: AI Applications Without a PhD](https://course.fast.ai/Resources/book.html). The examples on the book are not always the examples in the lessons. 

---

Materials for course sessions
  
  - **Session 'Introduction'**
    * Run [this notebook](test_GPU.ipynb) to compare CPU with CPU+GPU (try it in Colab: first, go to menu 'runtime/change runtime type' in Colab and choose GPU as hardware accelarator)
    
  - **Session 'Getting started'** (to do before February 24 class)
    - Watch video: Lesson 1 of [Practical Deep Learning for Coders 2022](https://course.fast.ai/) 
    - Notebooks adapted for Colab from [https://github.com/fastai/course22](https://github.com/fastai/course22):
      - [Lesson1_00_is_it_a_bird_creating_a_model_from_your_own_data.ipynb](Lesson1_00_is_it_a_bird_creating_a_model_from_your_own_data.ipynb), where one builds a classifier for images of birds and forests.
      - [Lesson1_02_saving_a_basic_fastai_model.ipynb](Lesson1_02_saving_a_basic_fastai_model.ipynb)
    - Notebook for Chapter 1 of the book [Deep Learning for Coders with Fastai and PyTorch](https://course.fast.ai/Resources/book.html)
      
  - **Session 'Deployment'** (to do before March 3rd class)
    - **Watch video**: Lesson 2 of [Practical Deep Learning for Coders 2022](https://course.fast.ai/) 
    - **Notebook**: Edited notebook for Chapter 2 of the book [Lesson2_edited_book_02_production.ipynb](Lesson2_edited_book_02_production.ipynb), where one builds and improve a classifier for bears.
    - **Deploy a fastai classifier on HuggingFace Spaces with Gradio** [https://www.tanishq.ai/blog/gradio_hf_spaces_tutorial](https://www.tanishq.ai/blog/gradio_hf_spaces_tutorial). 
    - To do that, you need to sign up first to [HuggingFace](https://huggingface.co/spaces)
    - You can also watch [this short video (8'53) that shows how I deployed the bear classifier example trained in Colab on HuggingFace Places using Gradio](https://www.youtube.com/watch?v=QkUyjwue3f4). Suggestion: in the colab notebook and `app.py`, comment (with #) or remove line of code `img = PILImage.create(img)` in the definition of `predict` since the output of `PILImage.create` misses some needed properties. 
    - The notebook [tests/read_model_launch_gradio_in_colab.ipynb](tests/read_model_launch_gradio_in_colab.ipynb) is similar to `app.py` and shows how you can test it first on Colab before you deploy it on HuggingFace Spaces.  
    
  - **Session 'Neural net foundations'** (to do before March 10th class)
    - **Watch video**: Lesson 3 of [Practical Deep Learning for Coders 2022](https://course.fast.ai/) **at least from ~23' to ~49'**.
    - **Colab notebook** for that part of the video: **How does a network really work?** (video from ~23' to ~49') [Lesson3_edited_04-how-does-a-neural-net-really-work.ipynb](Lesson3_edited_04-how-does-a-neural-net-really-work.ipynb). Uses a simple example (single variable function) to describe what we call the *loss*, what is *stochastic gradient descent* and how it can be implemented in Pytorch. It also describes and discusses the *ReLu* function.
           
  - **Session 'From-scratch model' (part 1)** (to do before March 17th: notice that Lesson 4 is postponed to a later date, so we move on to Lesson 5)
    - **Watch video**: Lesson 5 of [Practical Deep Learning for Coders 2022](https://course.fast.ai/), **at least up to 58'25''** (submitting results to Kaggle)
    - **Colab notebook** for the video: [Lesson5_edited_for_colab_linear_model_and_neural_net_from_scratch.ipynb](Lesson5_edited_for_colab_linear_model_and_neural_net_from_scratch.ipynb): follows the video up to 1:15'30'', but you only need to follow up to the end of the section 'Submitting results to Kaggle' (video ~58'25'').
    - In case you have trouble choosing the right Titanic data set from Kaggle, use train.csv from [titanic.zip](titanic.zip)
    - Start thinking about getting your own tabular data set for a classification problem. If you want, you can look at [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets) for data sets that you find interesting. 

  - **Session 'From-scratch model' (part 2)** (to do before March 24th)
    - **Watch the remainder part of the video**: Lesson 5 of [Practical Deep Learning for Coders 2022](https://course.fast.ai/)

  - **Session 'Natural Language (NLP)'**
    - Watch video: Lesson 4 of [Practical Deep Learning for Coders 2022](https://course.fast.ai/) 
    - Try the examples on the notebook [https://github.com/google/trax](https://github.com/google/trax): Run a pre-trained Transformer (English-German translator); Build a sentiment classification model from IMDB reviews.

---
Some other useful links:
- [fastai documentation](https://docs.fast.ai/)
- [AIquizzes](https://aiquizzes.com/)
- [Stanford Lecture Collection | Convolutional Neural Networks for Visual Recognition (2017)](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv): *Lesson 5* 9'07 (1st deep CNN diagram); 16' (input+kernel CNN); 21':23' (layers 1+2, activation maps, depth); 32'40 (stride); 34'54 (zero-pad); 45' (#parameters); 46' (common settings); 55'-56' (pooling layer); 1:01' (FC layer); 1:06' (demo url); 1:07'30 (summary, trend)
- [Notes for the Stanford course on Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)
- [Harvard CS50 | Introduction to Programming with Python free course](https://pll.harvard.edu/course/cs50s-introduction-programming-python)
- [Walk with Fastai free version tutorial](https://walkwithfastai.com/)
- [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)

---
Additional (optional) notebooks:
- Comparison of image models (video from ~14' ): [Lesson3_which_image_models_are_best.ipynb](Lesson3_which_image_models_are_best.ipynb)
- Edited notebook for Chapter 4 of the book [Lesson3_edited_book_04_mnist_basics.ipynb](Lesson3_edited_book_04_mnist_basics.ipynb). 
        - The first part of the notebook uses the MNIST data set (MNIST contains images of handwritten digits) and provides an introduction to *tensors* and to *PyTorch* in particular. This introduction includes a discussion about shapes and dimensions, loss (dissimilarity), broadcasting, etc. That first part of the notebook *is not* discused in the video. 
        - The second part of the notebook is about *Stochastic Gradient Descent* with some simple examples of one variable functions. 
