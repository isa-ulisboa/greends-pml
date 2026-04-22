# Practical Machine Learning, Green Data Science, 2nd semester 2025/2026

---
Instructor: Manuel Campagnolo, ISA/ULisboa (mlc@isa.ulisboa.pt)

Teaching assistant: Mekaela Stevenson (mekaela@edu.ulisboa.pt)

The course will follow a mixed flipped classroom model, where students are supposed to work on suggested topics autonomously before classes. Work outside class will be based on a range of Machine Learning resources including the book *Sebastian Raschka, Yuxi (Hayden) Liu, and Vahid Mirjalili. Machine Learning with PyTorch and Scikit-Learn. Packt Publishing, 2022*. During classes, Python notebooks will be typically run on Google Colab.

Links for class resources:
  - [Fenix webpage](https://fenix.isa.ulisboa.pt/courses/aaap-846413499992027). Course official page, where final results will be posted.
  - [Moodle ULisboa](https://elearning.ulisboa.pt/). Evaluation: assignments. The course is called [Practical Machine Learning](https://elearning.ulisboa.pt/course/view.php?id=10469). Students need to self-register in the Moodle page for the course.
  - [Kaggle](https://www.kaggle.com/). Access to data; candidate problems for the final project.

Some recommended tutorials:
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

- See (Raschka et al, 2022), Chapter 6: Learning Best Practices for Model Evaluation and hyper-parameter Tuning
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
<summary>  7. Combining Different Models for Ensemble Learning (April 24, 2025): random forest, gradient boosting, variable importance </summary>

- See (Raschka et al, 2022), Chapter 7:  Combining Different Models for Ensemble Learning

<!---


- See [Notes on ensemble learning and variable importance](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T5_ensemble_methods.md)
- Ensemble classifiers
- Random Forests
- Gradient boosting
- Exercise: adapt the classification pipeline to apply the XGBoost classifier (Montesinho burned area data set)
- Variable importance: MDI (Gini importance) and MDA (permutation importance)
- Pipeline that includes feature selection, followed by hyperparameter search: https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/wine_region_pipeline_XGB_CV_gridsearch_featselection.ipynb

</details>


<details markdown="block">
<summary> Backpropagation (Mar 7, 2025): SGD, forward pass, backward pass, PyTorch, optimizer, ... </summary>


- Video on the Perceptron and early times of AI [The First Neural Networks](https://www.youtube.com/watch?v=e5dVSygXbAE&t=88s)
- See (Raschka et al, 2022), Chapter 2: Training Simple Machine Learning Algorithms for Classification
- See [Basic concepts notes](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T1_basic_concepts.md). 
- Revise solutions for the problems listed in the previous class.
- Backpropagation and computation graph
- `PyTorch` pipeline: loss, optimizer
- The following table illustrates the changes from a basic Python script which is dependent on the model, loss, etc,  to a PyTorch higher-level script that can easily generalized to other models, loss functions or optimizer strategies.

| Basic Python | PyTorch 
|---|---
| Define model explicitly | Use a pre-defined model
|`def predict(x):`|`torch.nn.Linear(in_size,out_size)`
| Define loss explicitly | Use a pre-defined loss function
|`def loss(y,y_pred):`|`loss=torch.nn.MSEloss(y,y_pred)`
| Loss optimization strategy | Use a pre-defined optimizer
| Code explicitly| `optimizer=torch.optim.SGD(params, learn_rate)`
| Compute *ad hoc* gradient | **Use built-in backpropagation mechanism**
|`def gradient(x,y,y_pred):`|`loss.backward()`
|Update weights explicitly| `optimizer.step()`

</details>


<details markdown="block">
<summary> Data pipeline for deep learning  (May 9, 2025):  PyTorch, datasets, dataloaders</summary>

- See (Raschka et al, 2022), Chapter 12:   Parallelizing Neural Network Training with PyTorch
- See [Notebook on introduction to data pipelines for deep learning](https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/T6_pytorch_dataset_dataloader.ipynb). With deep learning (DL), it is possible to solve problems that involve complex input data like images, text and audio. The first step in order to apply DL is to organize the input data. PyTorch provides some key tools like `Dataset` and `DataLoader` that allow the creation of robust pipelines for DL.
- See [Veritasium video (3'42 to 14'50)](https://www.youtube.com/watch?v=GVsUOuSjvcg) for an historic introduction to multilayer neural networks  for deep learning.
- Run an interpret the code in pages 386-388 with an example of a dataset (`CelebA`) with several labels.
  
</details>


<details markdown="block">
<summary> Pipeline for deep learning with PyTorch (May 16, 2025):  data, model, model training and validation</summary>

- See (Raschka et al, 2022), Chapter 12: pp 389 to the end,  and Chapter 13: Going Deeper – The Mechanics of PyTorch, namely the MNIST project (ppp 436-439)
- See [Notebook the typical pipeline for deep learning with (non-convolutional) neural networks](https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/T7_torch_NN_pipeline.ipynb). In particular, we explore the MNIST dataset.
- Assignment #3 available on Moodle
- Suggestions of videos:
  - [PyTorch Course (2022), Part 4: Image Classification (MNIST)](https://www.youtube.com/watch?v=gBw0u_5u0qU)
  - [PyTorch Crash Course - Getting Started with Deep Learning](https://www.youtube.com/watch?v=OIenNRt2bjg)
  - [Build Your First Pytorch Model In Minutes! [Tutorial + Code](https://www.youtube.com/watch?v=tHL5STNJKag)
  - [MIT Introduction to Deep Learning 2025 (1:09)](https://www.youtube.com/watch?v=alfdI7S6wCY); Introduction up to "What is Deep Learning" (10'57); Why deep learning and why now (15'06); Building Neural Networks with Perceptrons (27'13); Applying NNs (35'30); Training NNs (41'21); NN in practice: Optimization (48'05).
    
</details>

<details markdown="block">
<summary> Deep convolutional neural networks  (May 23, 2025): input preparation, convolution, model architecture, receptive field </summary>

- See (Raschka et al, 2022), Chapter 14: Classifying Images with Deep Convolutional Neural Networks
- Check introductory video [What are CNNs?, by IBM (6'20)](https://www.youtube.com/watch?v=QzY57FaENXg)
- See [Notebook on introduction convolutional neural networks](https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/T9_CNNs_for_image_classification.ipynb). 
- Application of CNNs to the MNIST problem.
- Some techniques to improve deep learning: regularization, dropout, self-regularized activation functions, momentum, adaptive optimization. See https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/T8_techniques_to_improve_DP.ipynb
- Suggestions of videos:
  - [MIT 6.S191: Convolutional Neural Networks 2025 (1:01)](https://www.youtube.com/watch?v=oGpzWAlP5p0)
  
</details>

<details markdown="block">
<summary> Model deployment  (May 30, 2025):  saving and loading ML model, Gradio, Hugging Face places</summary>

- Saving and loading a PyTorch model. The following notebooks contain full pipelines to train a classifier for the MNIST dataset, including training (with `cuda` if available) and validation. The novelty is that we save the trained model after each epoch so it can be loaded later (for validation). This illustrates how a trained ML model can be saved to a file and loaded from a file, which is needed for deployment, fine-tuning and transfer learning.
  - https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/T9b_MNIST_CNNs_pipeline_save_load_model.ipynb : save the full model, which only works if the model is saved and loaded in the same device, which can be adequate for development in a local machine but is not recommended in general;
  - https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/T9c_MNIST_CNNs_pipeline_save_load_state_dict.ipynb : save only the model's learned parameters; it is the recommended way to save PyTorch models; to load, one first need to instantiate the model architecture and then load the weights.
  - https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/T9d_MNIST_CNNs_pipeline_save_load_jit_format.ipynb : JIT compilation provides a way to package your PyTorch model into a self-contained, optimized, and platform-independent format
- Deploying models with HF spaces.
    - Clone repository https://huggingface.co/spaces/mcampagnolo/test2024 to your local machine and run the app locally. Try making some changes (for instance, the messages to the user) on `app.py` and launch the app on your local machine to observe the changes. Note that the app uses a fine-tuned version of an adapted version (output size reduced from 1000 to 4) of a pre-trained `resnet18` model.
    - (optional) Choose a simple image classification app on Hugging Face spaces (e.g. https://huggingface.co/spaces/ByTixty1/Date_fruit-image-Classification/blob/main/app.py) and test it. Check the files `app.py`, `requirements.py`, `model.pth`. Try to understand the contents of `app.py` which runs Gradio and defines the interface.
- Improve the Gradio interface for the app you cloned
- Create your app in Hugging Face places: ideally you should build and test the app locally, and then push it to your HF space (she video below).
- Suggestions of videos:
  - [How to deploy a gradio app on huggingface (43')](https://www.youtube.com/watch?v=bN9WTxzLBRE&t=1845s)
  - [How to Create a Hugging Face Space: A Beginner's Guide (16')](https://www.youtube.com/watch?v=xqdTFyRdtjQ). Very clear video with a list of steps for creating HF space, creating basic files, testing on the local machine and pushing the Gradio interface into HF spaces. However, there are no details about the `app.py` code itself nor about the model that is deployed.
- Assignment #4: deploy a ML model on HF spaces (see Moodle)
  
</details>

<details markdown="block">
<summary> Foundation models and transfer learning  (June 6, 2025): types of ML problems and approaches, pre-trained models, fine-tuning</summary>

- Check introductory video [Machine Learning vs. Deep Learning vs. Foundation Models, by IBM (7'27)](https://www.youtube.com/watch?v=Beh13Cd_QbY).
- See (Raschka et al, 2022), search *fine-tuning* in Chapters 6, 11 and 16. Chapter 16 is the one where the concepts for this class are discussed in more detail. However, Chapter 16 deal with large language models (LLM) and the transformer architecture, which are not discussed in class. Nevertheless, the idea or using a pre-trained model (possibly a foundation model) and fine-tuning it is valid for any kind of model.
- See notes about foundation models, fine-tuning and transfer learning: https://github.com/isa-ulisboa/greends-pml/blob/maindocs/T10_ML_fine_tuning_transfer_foundation_models.md
- Notebook with the pipeline to load and adapt a pre-trained `resnet` model, freeze layers, and  fine-tune it:  https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/T10b_MNIST_resnet18_adapt_freeze_fine_tune.ipynb
- Try using a foundation Yolo model and high-level package from [Ultralytics](https://docs.ultralytics.com/models/). The [Yolov8](https://user-images.githubusercontent.com/27466624/212229562-003b8139-c8b5-4b0c-9d48-fe2f7b63243f.jpg) model is one of the available models for image tasks and be applied to different image sizes. Example of a notebook to fine-tune a `Yolov8n` model for grape leaf desease classification: https://colab.research.google.com/drive/1-kxX1kj6JzmFfyaXY4mMfq9EqWi34tww?usp=sharing (to fine-tune and predict, one needs to have data organized in folders in Google drive).
  
</details>


--- 

# Other resources

<details markdown="block">
<summary> Basic resources </summary>
  
- Sebastian Raschka, Yuxi (Hayden) Liu, and Vahid Mirjalili. Machine Learning with PyTorch and Scikit-Learn. Packt Publishing, 2022. See the presentation [webpage](https://sebastianraschka.com/blog/2022/ml-pytorch-book.html) and [GitHub repository](https://github.com/rasbt/machine-learning-book)
- [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)

</details>

<details markdown="block">
<summary> Tutorials </summary>
  
- [Machine Learning for Beginners (Microsoft)](https://microsoft.github.io/ML-For-Beginners/); [youtube channel](https://www.youtube.com/playlist?list=PLlrxD0HtieHjNnGcZ1TWzPjKYWgfXSiWG)
- [AI for Beginners (Microsoft)](https://microsoft.github.io/AI-For-Beginners/)
- [NYU course: Data Science for Everyone](https://www.youtube.com/@jonesrooy)
- [MIT 6.S191: Introduction to Deep Learning (2024)](https://www.youtube.com/watch?v=ErnWZxJovaM)
- [PyTorch tutorial by Patrick Loeber](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4). [Github repo](https://github.com/patrickloeber/pytorchTutorial)
- [Stanford Lecture Collection  Convolutional Neural Networks for Visual Recognition (2017)](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) and [Notes for the Stanford course on Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)
- [Stanford Machine Learning Full Course led by Andrew Ng (2020)](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU). Led by Andrew Ng, this course provides a broad introduction to machine learning and statistical pattern recognition. Topics include: supervised learning (generative/discriminative learning, parametric/non-parametric learning, neural networks, support vector machines); unsupervised learning (clustering, dimensionality reduction, kernel methods); learning theory (bias/variance tradeoffs, practical advice); reinforcement learning and adaptive control.
- [Broderick: Machine Learning, MIT 6.036 Fall 2020](https://www.youtube.com/watch?v=ZOiBe-nrmc4); [Full lecture information and slides](http://tamarabroderick.com/ml.html)
  
</details>
 



