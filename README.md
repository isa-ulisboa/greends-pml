# Practical Machine Learning, Green Data Science, 2nd semester 2025/2026

---
Instructor: Manuel Campagnolo, ISA/ULisboa (mlc@isa.ulisboa.pt)

Teaching assistant: Mekaela Stevenson (mekaela@edu.ulisboa.pt)

The course will follow a mixed flipped classroom model, where students are supposed to work on suggested topics autonomously before classes. Work outside class will be based on a range of Machine Learning resources including the book *Sebastian Raschka, Yuxi (Hayden) Liu, and Vahid Mirjalili. Machine Learning with PyTorch and Scikit-Learn. Packt Publishing, 2022*. During classes, Python notebooks will be typically run on Google Colab.

Links for class resources:
  - [Fenix webpage](https://fenix.isa.ulisboa.pt/courses/aaap-846413499992027). Course official page, where final results will be posted.
  - [Moodle ULisboa](https://elearning.ulisboa.pt/). Evaluation: assignments. The course is called [Practical Machine Learning](https://elearning.ulisboa.pt/course/view.php?id=10469). Students need to self-register in the Moodle page for the course.
  - [Kaggle](https://www.kaggle.com/). Access to data; candidate problems for the final project.

Some recommended video tutorials:
  - [Scikit-Learn ML basic tutorial](https://www.youtube.com/playlist?list=PLSE7WKf_qqo0lmPLmigvXDTDcKcGhre32). This covers the basics of using package `sklearn` for  ML, including basics, core terminology , linear regression, logistic regression, accuracy, preprocessing, and pipelines.
  - [Statistical Learning with Python - Stanford Online](https://www.youtube.com/playlist?list=PLoROMvodv4rPP6braWoRt5UCXYZ71GZIQ). This is an introductory-level course in supervised learning, with a focus on regression and classification methods. The lectures cover all the material in An Introduction to Statistical Learning, with Applications in Python by James, Witten, Hastie, Tibshirani and Taylor (Springer, 2023).
  - [MIT Introduction to Deep Learning](https://www.youtube.com/watch?v=alfdI7S6wCY)

<!---
[Overview notebook](https://github.com/isa-ulisboa/greends-pml/blob/main/ML_overview_with_examples.ipynb) This notebook provides an overview of the full course and contains pointers for other sources of relevant information and Python scripts.
--->
 
# Sessions
Each description below includes the summary of the topics covered in the session, as well as the description of assignments and links to videos or other materials that students should work through.

---

<details markdown="block">
<summary><a name="T0"></a> 0. Introduction (Feb 20, 2026) </summary>

We do an introduction to ML and compare it with *statistical modelling* using the simplest possible model, *linear regression*. We survey some of the problems that can be addressed with the techniques and tools that will be discussed during the semester. The examples will be run on Colab.

- See (Raschka et al, 2022), Chapter 1: Giving Computers the Ability to Learn from Data
- Types of machine learning problems: supervised learning, unsupervised learning, reinforcement learning, self-supervised learning, semi-supervised learning [comparison table](https://www.altexsoft.com/static/content-image/2026/1/self-supervised-learning-vs-other-major-machine-le-6013e.webp) Suggestion: check video [Types of machine learning](https://www.youtube.com/watch?v=gh6mNF2BGvk)
- Supervised learning: classification vs regression 
- Examples of input data for machine learning problems: tabular data, images, text.
- See *Iris data set* regression example with the notebook [iris_regression.ipynb](https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/iris_regression.ipynb). Identify the `.fit` and `.predict` methods, and explain what are their roles. Expand the code to perform some inference task, e.g. determine confidence bands for the responses, or determine confidence intervals for the regression coefficients.
- Statistics modeling  vs Machine Learning: Check video: [When to use stats vs. ML?](https://www.youtube.com/watch?v=xUsm34qnE30)
- The data set [Palmer Penguin](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data/data) has different type of variables and can be explored in different ways. The available attributes are:
    - species: penguin species (Chinstrap, Adélie, or Gentoo)
    - culmen_length_mm: culmen length (mm)
    - culmen_depth_mm: culmen depth (mm)
    - flipper_length_mm: flipper length (mm)
    - body_mass_g: body mass (g)
    - island: island name (Dream, Torgersen, or Biscoe) in the Palmer Archipelago (Antarctica)
    - sex: penguin sex <br>
  Try to understand the data and establish and explore a meaningful regression problem using a statistical and a ML approach. For example, try to predict the body mass using as predictors the culmen length and depth and the flipper length (regression problem).
<!--- An example of a prediction task for time series: check the notebook [modeling ground water levels](https://www.kaggle.com/code/andreshg/timeseries-analysis-a-complete-guide/) for the Kaggle competition [Acea Smart Water Analytics](https://www.kaggle.com/competitions/acea-water-prediction/). Try to download the data and run the notebook to reproduce the results. --->
</details>

---

<details id= markdown="block">
<summary><a name="T1"></a> 1. Basic concepts (Feb 27, 2026): model, loss, fit, learning rate, iterations, epochs </summary>

The goal of the following classes is to understand how ML models can be trained in and used to solve regression and classification problems. We start by applying the machine learning approach to well-known statistical models like linear regression to illustrate the stepwise approach followed in ML. We extend the approach to binary classification problems. 

- Presentation of the 1st assignment on Moodle (due date: March 11, 2026)
- See (Raschka et al, 2022), Chapter 2: Training Simple Machine Learning Algorithms for Classification
- Check the [introductory video on LR](https://www.youtube.com/watch?v=3dhcmeOTZ_Q) using ML approach.
- Basic concepts in Machine learning: *model*, *fit*, *epochs*, *loss*, *learning rate*, *weights*, for a simple regression problem. See [Basic concepts notes](docs/T1_basic_concepts.md).
- Exercise: define a Linear Regression class to with methods `.fit`and `.predict` and visualize the iterative process to find the optimal weights. See [exercise](docs/T1_Create_LInearRegression_class_fit_predict_visualize.md).
- [Exercise](docs/T1_linear_regression_exercise_with_pseudo_code.md): consider the pseudo-code for the previous exercise, and relate it to concepts *loss function* and *stochastic gradient descent*.
- Extend the optimization approach to a binary classification problem. See [Basic concepts notes](docs/T1_basic_concepts.md).
</details>

---

<details markdown="block">
<summary><a name="T2"></a> 2. Basic concepts (Mar 6, 2026): Classification, logistic regression, entropy and cross-entropy, regularization, batch size</summary>

- See (Raschka et al, 2022), Chapter 3, pp 59-76
- See [Basic concepts notes](docs/T2_basic_concepts_classification.md).
- Check this very basic description of [Logistic Regression](https://www.youtube.com/watch?v=OlKL5nzm-1w)
- Entropy, cross-entropy and KL Divergence: check video on [KL divergence 0-4'20](https://www.youtube.com/watch?v=tXE23653JrU) and [A Short Introduction to Entropy, Cross-Entropy and KL-Divergence, with application to ML](https://www.youtube.com/watch?v=ErfnhcEV1O8)
- Exercise (part 1): Download the data set [Breast Cancer Wisconsin data set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). Read and explore data, split in train and test, standardize data, build from scratch a `LogisticRegression class`, use training data to fit the model, evaluate the accuracy with test data. Finally, plot the fitted regression with the linear combination of predictors on the *x* axis, and the probabilities on the *y* axis; the plot should depict the malignant cases and the benign cases in different colors  
- Exercise (part 2): adapt the `LogisticRegression` class so you can process training data in batches;
- Exercise (part 3): adapt it further to include a regularization term in the loss function.

  <details markdown="block">
  <summary>Suggestion for the script (to be completed)</summary>
    
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    # from your_module import LogisticRegression  <-- to be implemented
    
    def main():
        # 1. Load and Clean Data
        df = pd.read_csv('data.csv')
        # Drop unnecessary columns and encode target (M=1, B=0)
        df = df.drop(['id', 'Unnamed: 32'], axis=1)
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        
        X = df.drop('diagnosis', axis=1).values
        y = df['diagnosis'].values
    
        # 2. Split and Standardize
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
        # 3. Build and Fit Model (Exercises 1, 2, & 3)
        model = LogisticRegression(
            learning_rate=0.01, 
            epochs=1000, 
            batch_size=32,      # Part 2
            lambda_reg=0.1      # Part 3
        )
        model.fit(X_train_scaled, y_train)
    
        # 4. Evaluation
        predictions = model.predict(X_test_scaled)
        accuracy = (predictions == y_test).mean()
        print(f"Model Accuracy: {accuracy * 100:.2%}")
    
        # 5. Visualization
        # Calculate linear combination (z = Xw + b) and probabilities
        z = X_test_scaled @ model.weights + model.bias
        probs = model.predict_proba(X_test_scaled) 
    
        plt.figure(figsize=(10, 6))
        plt.scatter(z[y_test == 1], probs[y_test == 1], color='red', label='Malignant', alpha=0.5)
        plt.scatter(z[y_test == 0], probs[y_test == 0], color='blue', label='Benign', alpha=0.5)
        
        # Plot the sigmoid curve
        plt.title("Logistic Regression: Linear Combination vs Probability")
        plt.xlabel("Linear Combination (z)")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    if __name__ == "__main__":
        main()
      
    ```
  </details>
  
</details>

---

<details markdown="block">
<summary> 3. Decision trees (Mar 13, 2026): decision trees for classification, information gain, over-fitting, train and development sets </summary>

- Review structured code for the exercise of the previous class on [Breast Cancer Wisconsin data set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).
    - Pipeline: train/test + pre-processing (scale features) + define model + fit Model + use model to predict;
    - Other concepts: logistic regression; batch size, regularization parameter.
- Discussion of [Assignment #1](notebooks/assign_1_wine_quality.ipynb). Keywords: input, output, model, loss function, epoch, batch, predict, train dataset, and (independent) test dataset.
- See (Raschka et al, 2022), Chapter 3: Decision tree learning (pg 86-98)
- See [Decision tree notes](docs/T3_decision_trees_overfitting_train_dev.md)
- Check this video for an easy introduction to decision trees using `sklearn.tree.DecisionTreeClassifier`: [Pokemon classifier](https://www.youtube.com/watch?v=LLBGiAAZqAM)
- The risk of over-fitting: train and development (validation) data sets
- Decision tree hyper-parameters, e.g. `max_depth`
- Exercise: create a decision tree classifier for the [Soil detection for cotton crop problem](https://www.kaggle.com/datasets/zohasohail/soil-detection-for-cotton-crop). Use as predictors `['ph', 'Temperature', 'Humidity', 'Density', 'Electrical Conductivity', 'N', 'P', 'K']` and as response `'Cotton Crop'`. Determine the best values for hyper-parameters Maximum depth and Minimum leaf size using a development (validation) set. Visualize the model with `plot_tree`. See [possible structure for the code](notebooks/T3_cotton_crop_problem_grid_search.ipynb). Note that `sklearn.tree.DecisionTreeClassifier` can only be applied to numerical features. If categorical features are available, they must be converted to numerical (typically using a one-hot encoder).
- Comparision of logistic regression with decision trees for classification:

| Model | Logistic Regression | Decision tree |
| --- | --- | ---|
| Problem | Classification | Classification |
| Hyper-parameters | learning rate, number iterations, ... | tree depth, leaf size, ... |
| Risk of over-fitting | low | high |
| Loss function | cross entropy: $-\log\_2\hat{p}\_i$, $i$ is the actual label | Gini, or entropy:  $-\sum_{i=1}^n\hat{p}\_i\log\_2\hat{p}\_i$ |
| Optimization | Gradient descent | Brute force (try all features and all thresholds) |
  
</details>

---


<details markdown="block">
<summary> 4. Data preprocessing (Mar 20, 2026): pipelines, missing data, categorical features, scaling</summary>

- See (Raschka et al, 2022), Chapter 4 (Data Preprocessing) and Chapter 6 (Streamlining workflows with pipelines)
- See [Scikit-Learn ML basic tutorial](https://www.youtube.com/playlist?list=PLSE7WKf_qqo0lmPLmigvXDTDcKcGhre32), sections 12, 13, 14 and 15.
- Supervised learning flowchart
  <details markdown="block">
  <summary>Figure 1.9 (Raschka et al, 2022) </summary>
  <img src="https://github.com/isa-ulisboa/greends-pml/blob/main/docs/supervised_learning_flowchart_raschka_2022.png" alt="Alt Text" width="600" >
  </details>
- The Titanic data set example: See [Pre-processing notes](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T4_missing_data_categorical_scaling.md)
- Removing and imputing missing values from the data set
- Handling categorical data;
- Bringing features onto the same scale;
- Partitioning a dataset into separate training and test datasets;
- Scikit learn pipeline: `.transform`, `.fit` and `.predict` methods.
  <details markdown="block">
  <summary>Figure 6.1 (Raschka et al, 2022) </summary>
  <img src="https://github.com/isa-ulisboa/greends-pml/blob/main/docs/pipeline_fig_6_1.png" alt="Alt Text" width="500">
  </details>
- Exercise: apply the principles and code discussed above to the Montesinho burned area data set. You can convert the problem into a classification problem by categorizing the original response variable (burned area). See [Pre-processing notes](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T3_missing_data_categorical_scaling.md)
</details>

---

<details markdown="block">
<summary>5. Model Evaluation and hyper-parameter Tuning (Mar 27, 2026): cross-validation, strata and groups, grid-search </summary>

- See (Raschka et al, 2022), Chapter 6: Learning Best Practices for Model Evaluation and hyper-parameter tuning
- Discussion of assignment \#2
- See [Cross-validation and hyper-parameter tuning notes](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T5_cross_validation.md)
- Check video [Complete guide to cross-validation](https://www.google.com/search?client=firefox-b-d&hs=M8F&sa=X&sca_esv=ea79f29eab3dedca&sxsrf=ANbL-n52glIguINYT9mbWgCjy-ZXtaWjdA:1774536555619&uds=ALYpb_ncDc7jTlmw6Mmq7NjuX5c-Uy1yO4MtdEOyw56oQr4pD_xy9m9pUVOBFZgMYXBhoTkwcXjEVdjxilFCaKGaLRAsUSY7tYUnSuHswSwuSw_nQtstn67jn2dndqdLjdJqSsMbrfWlU84G5ZyyLRLuzVGbW-9LLuv7Kzh4BbLjrscozO5zF7IvkIOpYmvtpKowhIl1BVkcGzMW-SqCwtcoKLNPM3XrHgXAapOSBT9p3IE78H-RrEA&udm=7&fbs=ADc_l-aN0CWEZBOHjofHoaMMDiKpmAsnXCN5UBx17opt8eaTX5MJRoosnbembaWTjeNSquIxro2mrW6zffXrbXZY-opPXGY0Rt_bdDSE237xSnWdKR3dIcuWpVYnCh4I-6IiMCln65mNNN2yH1ysO3lP5K7J78yX6_da8m1AE3qAXevBHCVFtwF3sLVw9ZzZFWqV0P01yhOM&q=Cross+validation+with+sklearn+tutorial&ved=2ahUKEwiloafo572TAxXXxgIHHb9KCtsQtKgLegQIEBAB&biw=1536&bih=769&dpr=2.5#fpstate=ive&ip=1&vld=cid:077c95e2,vid:-8s9KuNo5SA,st:0) 12:40-end. Note that the dataset used for this tutorial is the "Stroke prediction data" where one of the features is the patient's doctor. This information is important because individuals should be *grouped* according to this feature.
- Streamlining workflows with pipelines
- Using k-fold cross-validation to assess model performance
- Debugging algorithms with learning and validation curves
- Fine-tuning machine learning models via grid search
</details>

---

<details markdown="block">
<summary> 6. Evaluation metrics (Apr 10, 2026): confusion matrix, precision, recall, F1-score, ROC curve, AUC </summary>

- See (Raschka et al, 2022), Chapter 6: Learning Best Practices for Model Evaluation and hyper-parameter Tuning
- Revise pipelines, train vs test -- see diagram (https://github.com/isa-ulisboa/greends-pml/blob/main/docs/holdout_method_fig62.png)
- See notes on  [Evaluating performance](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T6_evaluate_accuracy.md)
- Complete exercise [spanish white wines high ratings](https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/spanish_white_wine_high_ratings.ipynb)
- Suggestion: check video on [Evaluation Metrics For Classification - Full Overview](https://www.youtube.com/watch?v=pGPiRRfNsr0)
- See [binary classification metrics table](https://en.wikipedia.org/wiki/Template:Diagnostic_testing_diagram): confusion matrix and derived metrics
- Receiver operating characteristic curve, also known as relative operating characteristic curve (ROC) and precion-recall curves
- Scoring metrics for multiclass classification 
- Dealing with class imbalance

</details>

---

<details markdown="block">
<summary>  7. Combining Different Models for Ensemble Learning (April 24, 2026): random forest, gradient boosting, variable importance </summary>

- See (Raschka et al, 2022), Chapter 7:  Combining Different Models for Ensemble Learning
- See [Notes and examples on ensemble learning and variable importance](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T7_ensemble_methods.md)
- Ensemble classifiers
- Random Forests
- Gradient boosting and XGBoost
- Exercise: adapt the classification pipeline to apply the XGBoost classifier (Montesinho burned area data set)
- Variable importance: MDI (Gini importance) and  MDA (permutation importance) for Random Forest; SHAP for generic ML models. Check video [Explainable AI explained! | #4 SHAP](https://www.youtube.com/watch?v=9haIOplEIGM) for a short and clear explanation on how SHAP works. The SHAP package provides many useful visualizations tools.
  
<!--- - Pipeline that includes feature selection, followed by hyperparameter search: https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/wine_region_pipeline_XGB_CV_gridsearch_featselection.ipynb --->

</details>

---

<details markdown="block">
<summary> 8. Introduction to deep learning; data pipeline with PyTorch (May 8, 2026):  datasets, dataloaders</summary>

- Videos on the history of neural networks: Video on the Perceptron and early times of AI [The First Neural Networks](https://www.youtube.com/watch?v=e5dVSygXbAE&t=88s); Video on [ChatGPT is made from 100 million of these [The Perceptron]](https://www.youtube.com/watch?v=l-9ALe3U-Fg)
- See (Raschka et al, 2022), Chapter 2: Training Simple Machine Learning Algorithms for Classification; review [Basic concepts notes](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T1_basic_concepts.md) for some fundamentals about ML that also apply to deep learning, and also Chapter 12: Parallelizing Network Training with PyTorch.
- Introduction to neural networks. See video [But what is a neural network? | Deep learning chapter 1, from 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) on deep learning with an example of a fully connected neural network with two hidden layers for handwritten digit recognition.
- General `PyTorch` pipeline: see [illustration of a PyTorch workflow](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/01_a_pytorch_workflow.png)
- See [Notebook on introduction to data pipelines for deep learning](https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/T8_pytorch_dataset_dataloader.ipynb). With deep learning (DL), it is possible to solve problems that involve complex input data like images, text and audio. The first step in order to apply DL is to organize the input data. PyTorch provides some key tools like `Dataset` and `DataLoader` that allow the creation of robust pipelines for DL.
- Assignment 3, available on Moodle. Explore in particular the concepts of `Dataset` and `Dataloader` in that notebook.

</details>

---

<details markdown="block">
<summary> 9. Neural networks (May 15, 2026): backpropagation, gradient descent, forward pass, backward pass,  optimizer, ... </summary>
  
- See (Raschka et al, 2022), Chapter 12: pp 389 to the end,  and Chapter 13: Going Deeper – The Mechanics of PyTorch, namely the MNIST project (ppp 436-439)
- In Google Colab or some other platform, prompt the AI bot with "Create a PyTorch script to train a fully connected neural network for the MNIST data set". Analyze the proposed script.
- See [Notebook the typical pipeline for deep learning with (non-convolutional) neural networks](https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/T9_torch_NN_pipeline_MNIST88.ipynb) that explores and discusses a PyTorch pipeline for applying a (non convolutional) NN to the MNIST data set and validate results.
- Video suggestions:
  - [PyTorch Crash Course - Getting Started with Deep Learning (50')](https://www.youtube.com/watch?v=OIenNRt2bjg). Detailed discussion of PyTorch Deep Learning framework with simple examples.
00:00 Intro & Overview
00:54 Installation & Overview; 
02:37 Tensor Basics; 
11:14 Autograd; 
17:41 Linear Regression Autograd; 
20:59 Model, Loss & Optimizer; 
27:11 Neural Network; 
38:08 Convolutional Neural Net. Application: image classification CIFAR10 data set (objects color images).
  
  - [Build Your First Pytorch Model In Minutes: Tutorial + Code](https://www.youtube.com/watch?v=tHL5STNJKag).
    00:00 Intro; 
04:50 Pytorch Datasets; 
13:59 Pytorch Model (EfficientNet, with added FC layer); 
19:19 Pytorch Training; 
29:23 Results. The application is image classification (recognize playing cards). Note that this example illustrates the idea of *transfer learning*.

  - [PyTorch basic concepts](https://www.youtube.com/watch?v=r1bquDz5GGA). This video describes basic concepts and constructs from PyTorch in a clear way. 
0:00:00 - Introduction
0:01:55 - torch.tensor; 
0:07:32 - Autograd & requires_grad; 
0:09:21 - Computation Graph & .grad_fn; 
0:12:50 - Element-wise vs. Matrix Multiplication; 
0:15:25 - Reduction Operations & The dim Argument; 
0:30:25 - loss.backward(); 
0:34:12 - Gradient Descent Update Rule; 
0:40:23 - torch.nn.Module; 
0:52:19 - torch.optim; 
0:57:30 - Transformer Feed-Forward Network; 

- Notes on [Computing gradients in PyTorch](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T8_computing_gradients_with_pytorch.md) with links to videos on *Backpropagation* and  *Gradient Descent with Autograd and Backpropagation*.
  
</details>

---

<details markdown="block">
<summary> 10. Deep convolutional neural networks  (May 22, 2026): convolution, model architecture, encoder and decoder </summary>

- See (Raschka et al, 2022), Chapter 14: Classifying Images with Deep Convolutional Neural Networks
- Check introductory video [What are CNNs?, by IBM (6'20)](https://www.youtube.com/watch?v=QzY57FaENXg)
- Example: very compact notebook for [Grape disease identification](https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/T10_CNN_Grape_Disease_Identification_Yolo8n.ipynb). The goal is to train a convolutional neural network (CNN) to identify if the plant is healthy or has  Black Rot, ESCA, Leaf Blight, using data from the [Grapevine Disease Dataset](https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original) and a pre-trained YOLO model. We see how to fine-tune the model from data already saved on Google Drive and interpret the model and the results.
- Notebook on [Convolutional NNs for image classification](https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/T10_CNNs_for_image_classification.ipynb). This notenook focus on some of the main parameters for CNNs: convolutions and kernels, padding, pooling, stride, activation map, receptive field, batch normalization and dropout.
- CNNs for image identification, image detection (e.g. YOLO, R-CNNs) and image segmentation (e.g. U-nets): encoders and decoders. Check video [MIT 6.S191: Convolutional Neural Networks](https://www.youtube.com/watch?v=pqIcoskUuWs&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI).
- In Google Colab or another IDE with an AI bot, prompt "Create a PyTorch script from that shows how to fine tune a custom U-net for image segmentation" and analyze the proposed notebook.
- Autoencoders: this is a very powerful technique that can be implemented with neural networks. Check the example for MNIST with a NN and a CNN at [Autoencoder In PyTorch - Theory & Implementation](https://www.youtube.com/watch?v=zp8clK9yCro). That concept supports most generative AI: check the overview video on [MIT 6.S191: Deep Generative Modeling](https://www.youtube.com/watch?v=R8V8CbuxryI&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=4).

</details>

---

<details markdown="block">
<summary> 11. Model deployment  (May 29, 2026):  saving and loading ML model, Gradio, Hugging Face places</summary>

- A PyTorch learning model has two parts:
  - The *model architecture* that includes layers, activation functions, forward pass. This is stored as **Python code**;
  - The learned weights, which are numbers. This is stored in a `.pth` file.

 - Typical pipeline for creating and deploying a model:
   - Create model, train it with reference data and save weights with `torch.save(model.state_dict(), "model_weights.pth")`. This could be done, e.g. in Colab or in your own PC.
   - Re-create the model, load the weights, and predict over new examples. This could be done e.g. on a platform like Hugging Face spaces (see below)
   - The only data you need to load to deploy the model is the `model_weights.pth` file.

- Notebook that illustrates how to create a simple deep learning model, train it and save the trained weights and, then, re-create the model, load the weights and use it for prediction: <https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/T11_save_load_apply_model.ipynb>.
- Deploying models with HF spaces.
    - Check the app on Hugging Face spaces  <https://huggingface.co/spaces/mcampagnolo/test2024/blob/main/app.py> to apply the prediction model for Grapevine Disease Dataset we discussed last class and test it. Check the files `app.py`, `requirements.py`, and `model.pth`: what is they for? Try to understand the contents of `app.py` which runs Gradio and defines the interface. 
- You can download the app and improve it, in particular you can improve the Gradio interface for the app.
- Assignment #4: create your own app in Hugging Face places
- Suggestions of videos:
  - [How to deploy a gradio app on huggingface (43')](https://www.youtube.com/watch?v=bN9WTxzLBRE&t=1845s)
  - [How to Create a Hugging Face Space: A Beginner's Guide (16')](https://www.youtube.com/watch?v=xqdTFyRdtjQ). Very clear video with a list of steps for creating HF space, creating basic files, testing on the local machine and pushing the Gradio interface into HF spaces. However, there are no details about the `app.py` code itself nor about the model that is deployed.
  
</details>

--- 

<details markdown="block">
<summary> 12. Foundation models and transfer learning  (June 5, 2026): types of ML problems and approaches, pre-trained models, fine-tuning</summary>

-  Textbook (Raschka et al, 2022): search *fine-tuning* in Chapters 6, 11 and 16. Chapter 16 is the one where the concepts for this class are discussed in more detail. However, Chapter 16 deals with large language models (LLM) and the transformer architecture, which has not been discussed in detail in class. Nevertheless, the idea or using a pre-trained model (possibly a foundation model) and fine-tuning it is valid for any kind of model.
- Check introductory video [Machine Learning vs. Deep Learning vs. Foundation Models, by IBM (7'27)](https://www.youtube.com/watch?v=Beh13Cd_QbY).
- See video on [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M). For more details, check for instance [Large Language Models explained briefly, by 3Blue1Brown](https://www.youtube.com/watch?v=LPZh9BOjkQs)
- There are foundation models for specific areas of application. As an example, search for "Geospatial" in Hugging Face to list of available foundational models that have been trained with satellite imagery. For the same area of application, check [Google's Satellite Embedding V1](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) which provide ready to use deep features extracted from Earth observations for a wide range of earth observation applications.  
- See notes about foundation models, fine-tuning and transfer learning: https://github.com/isa-ulisboa/greends-pml/blob/maindocs/T10_ML_fine_tuning_transfer_foundation_models.md
- Notebook with the pipeline to load and adapt a pre-trained `resnet` model, freeze layers, and  fine-tune it:  https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/T10b_MNIST_resnet18_adapt_freeze_fine_tune.ipynb

</details>

