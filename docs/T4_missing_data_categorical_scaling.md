# Pre-processing for train and test sets

In these notes, we discuss several pre-processing typical steps and show how they can be organized as transformations within a structured pipeline that handles train and test data. Train data are subjected to `.fit`, `.transform` and `.predict` methods while test data are only subjected to `.transform` and `.predict`.

---
    
</details>

<details markdown="block">
<summary>  Missing values </summary>

## Missing values

Some traditional machine learning tools are designed to handle missing values: [sklearn Estimators that handle NaN values](https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values). To understand how this is done, ones needs to read the documentation.

There are some typical approaches:
- Replace missing values by a constant;
- Replace missing values by the mean or median of the known values (numerical variables)
- Replace missing values by the most frequent value

See [Titanic missing data notebook](https://github.com/isa-ulisboa/greends-pml/tree/main/notebooks/titanic_missing_data.ipynb)

---
    
</details>

<details markdown="block">
<summary>Handling categorical data</summary>

## Handling categorical data

Categorical data can be or not ordinal (see for example "class" and "embark_town" in the Titanic example). Ordinal data can be converted into one single numerical variable. However, non ordinal variables are typically converted into a collection of binary variables, with the possibility of dropping one to avoid redundacy.

This can be done easily with `pandas.get_dummies`: see [Titanic get dummies notebook](https://github.com/isa-ulisboa/greends-pml/tree/main/notebooks/titanic_get_dummies.ipynb)

---
    
</details>

<details markdown="block">
<summary>Bringing features onto the same scale</summary>


## Bringing features onto the same scale

There are many strategies to scale different numerical attributes. The most common are:
- Rescale to a common interval, e.g. [0,1] using minimum and maximum
- Standardize using mean and variance
Note: in this context, the `.fit` method is where the minimum, maximum, mean, standard deviation, etc, are computed. Therefore, it should only be applied to the train data set. The `.transform` method that uses those calculated parameters is identical for train and test data.

**Script**: See [Titanic scaling notebook](https://github.com/isa-ulisboa/greends-pml/tree/main/notebooks/titanic_scaling.ipynb)

---
    
</details>

<details markdown="block">
<summary>Pre-processing with a pipeline to avoid leakage between train and test </summary>

## Pre-processing with a pipeline to avoid leakage between train and test

Writing code "by hand" might lead to errors and leakage in particular when using test data to estimate precision. Pipelines are structured to help avoiding those problems: see [Scikit learn pipelines](https://scikit-learn.org/stable/modules/compose.html#pipeline). Check the blog on ["How to apply preprocessing steps in a pipeline only to specific features"](https://medium.com/analytics-vidhya/how-to-apply-preprocessing-steps-in-a-pipeline-only-to-specific-features-4e91fe45dfb8)

**Script**: See a basic example of a pipeline for pre-processing data and for applying correctly methods `.transform`, `.fit` and `.predict` to train and test data: [Titanic pre-processing pipeline notebook](https://github.com/isa-ulisboa/greends-pml/tree/main/notebooks/titanic_preprocessing_pipeline.ipynb)

The core part of the script is the construction of the pipeline. This includes pre-processing pipelines for categorical and for numerical attributes. Then, `ColumnTransformer` applies those transformations to the data, with `remainder = 'drop'` to drop the additional attributes. Finally, `Pipeline` combines the pre-processing and classification steps. The remainder of the script follows the standard procedure that  includes `pipeline.fit(X_train, y_train)` and `y_pred=pipeline.predict(X_test)`.

    categorical_features = ['pclass', 'sex', 'embarked']
    categorical_transformer = Pipeline(
        [
            # ('imputer_cat', SimpleImputer(strategy = 'constant', fill_value = 'missing')),
            ('imputer_cat', SimpleImputer(strategy = 'most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
        ]
    )
    numeric_features = ['age', 'sibsp', 'parch', 'fare']
    numeric_transformer = Pipeline(
        [
            ('imputer_num', SimpleImputer(strategy = 'median')),
            #('scaler', StandardScaler())
            ('normalizer', Normalizer())
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ('categoricals', categorical_transformer, categorical_features),
            ('numericals', numeric_transformer, numeric_features)
        ],
        remainder = 'drop' # By default, only the specified columns in transformers are transformed and combined in the output, and the non-specified columns are dropped.
    )
    pipeline = Pipeline(
        [
            ('preprocessing', preprocessor),
            ('clf', RandomForestClassifier(n_estimators=10)) #tree.DecisionTreeClassifier()) # LogisticRegression())
        ]
    )
     
Note: the script above uses class `Pipeline`. One alternative that makes the code shorter is to use `make_pipeline` (https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html). This is a shorthand for the Pipeline constructor; it does not require, and does not permit, naming the estimators. Instead, their names will be set to the lowercase of their types automatically.

---
    
</details>

<details markdown="block">
<summary> Exercise: Montesinho burned area data set (with numerical and categorical variables) </summary>
  
## Exercise: Montesinho burned area data set (with numerical and categorical variables)

Consider the data set that describes 517 fires from the Montesinho natural park in Portugal. For each incident weekday, month, coordinates, and the burnt area are recorded, as well as several meteorological data such as rain, temperature, humidity, and wind (https://www.kaggle.com/datasets/vikasukani/forest-firearea-datasets). The variables are:

- X - x-axis spatial coordinate within the Montesinho park map: 1 to 9
- Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9
- month - month of the year: "jan" to "dec"
- day - day of the week: "mon" to "sun"
- FFMC - FFMC index from the FWI system: 18.7 to 96.20
- DMC - DMC index from the FWI system: 1.1 to 291.3
- DC - DC index from the FWI system: 7.9 to 860.6
- ISI - ISI index from the FWI system: 0.0 to 56.10
- temp - the temperature in Celsius degrees: 2.2 to 33.30
- RH - relative humidity in %: 15.0 to 100
- wind - wind speed in km/h: 0.40 to 9.40
- rain - outside rain in mm/m2 : 0.0 to 6.4
- area - the burned area of the forest (in ha): 0.00 to 1090.84. IN fact we are going to convert this into a binary variable by asking if fires are larger than 5 ha

### Read data

- Read the data and create arrays `X`and `y`. You can discard the original grid coordinates `X,Y` and just keep attributes 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain' and the response variable 'area'.
- Try to fit a regression tree to the data. Do you get the error message `could not convert string to float: 'mar'`?

### Categorical variables and `get_dummies`
- Solve that problem with `pd.get_dummies`. What does this do?
- What happens if you use the `drop_first=True` option?
- And the `dtype=float` option?

### Set up pipeline for pre-processing and train/test

See the following Scikit-learn documentation:

- [Pipelines](https://scikit-learn.org/stable/modules/compose.html#pipeline)
- [ColumnTransformer for heterogeneous data](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data)

</details>
