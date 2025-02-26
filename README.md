## Iris Detector: the famous Iris flowers dataset
This project sources data from the iris data set: https://archive.ics.uci.edu/dataset/53/iris
We can follow the machine learning workflow we described.

Other sources of reference for this: 
- https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
- https://www.kaggle.com/code/mohamedtarek111/iris-classification-with-100-acc/notebook 

## 1. Data Collection 
Some useful questions to ask when collecting data is:
- What data is relevant?
- How was the data sampled?
- How can we clean the data?
- Are there privacy issues?

## 2. Data preparation
Once we've loaded the data in, it's good to look at the data and understand the data from multiple angles by describing using stats and visualising the data.

### Visualising the data
There are two typs of plots we are going to look at: univariate and multivariate. 
1. Univariate look at relationship of one variable. Examples: histograms, bar charts, pie charts and box plots.
2. Multivariate look at relationships between two or more variables. Example: scatter plot matrix

### Split the data
Split the data into a training set and testing set. The training set is used to train the model and the remaining 20% is used as test data. This test data will be used in the validation stage to see how accurate the model is. It's unclear which algorithmn to use for this dataset so we should try different algorithm and evaluate which one is best.

Linear:
- Logistic Regression (LR): models chances of an instance being in a class by using a line with the logistic sigmoid function.
    - **Anaology**: Imagine you have two types of fruits, apples and oranges, and you want to separate them based on their size and color. Logistic Regression draws an imaginary line that divides the fruits into two groups, apples on one side and oranges on the other 

- Linear Discriminant Analysis (LDA): finds the best direction that separates the classes and projects the data onto that direction.
    - **Anaology**: Let's say you have a bunch of toys, like cars, dolls, and balls, and you want to sort them into different boxes. LDA finds the best way to tilt the boxes so that when you drop the toys, they separate into the correct boxes based on their shapes and sizes 

Non-linear: 
- K-Nearest Neighbors (KNN): finds the closest neighbours (most similar)
    - **Anaology**: Imagine you have a new toy, and you want to know if it's a car, a doll, or a ball. KNN looks at the toys that are most similar to your new toy (the K nearest neighbors) and puts your new toy in the same group as the majority of those similar toys.
- Classification and Regression Trees (CART).
- Gaussian Naive Bayes (NB).
- Support Vector Machines (SVM)

We will split the data into 10 parts where the data is trained on 9 parts and tested on 1. Stratified means that each split has the same distribution of examples by class. This helps prevent any single fold from being biased towards certain classes and ensures that each fold is representative of the overall data distribution.

The random_state argument is set to a fixed number to ensure that each algorithm is evaluated on the same splits of the training dataset. This ensures reproducibility and a fair comparison between different machine learning algorithms, it is important that all algorithms are evaluated on the same splits of the training data. The random_state argument is a parameter that allows you to set the starting point (seed) for the random number generator used in the shuffling and splitting process. This will produce the same sequence of random numbers every time, resulting in the same shuffling and splitting of the dataset.

The scoring argument is set to ‘accuracy‘ to evaluate models. This is a ratio of the number of correctly predicted instances divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate)

## 4. Evaluate the model
Compare the models.

## 5. Deploy the model
The resutls of the LinearDiscriminantAnalysis look strange, so the next best model SVC is selected and used to make the prediction.