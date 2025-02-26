import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

# Import iris flower dataset
columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class'] 
df_iris=pd.read_csv('iris/iris.data',names=columns)

print(df_iris.describe())

# Check for any null values
df_iris.isnull().sum()

# This tells us how many flowers are in each class
df_iris.value_counts('class')

# box and whisker plot
df_iris.plot(kind='box',layout=(2,2))
plt.savefig('plots/box_whisker.png')

# scatter plot matrix
scatter_matrix(df_iris)
plt.savefig('plots/scatter.png')
plt.clf()

# Devide data into  
X=df_iris.iloc[:,:-1]
y=df_iris.iloc[:,-1]

#Splitting the data into the training and testing set  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10) 

models = []
models.append(('LogisticRegression', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
models.append(('GaussianNB', GaussianNB()))
models.append(('SVC', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
	
# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/algo_comparisons.png')

# creates an instance of the Support Vector Classifier (SVC) model 
model = SVC(gamma='auto')
# This line trains the SVC model using the training data
model.fit(X_train, y_train)
# The trained model is used to make predictions on the test data X_test
predictions = model.predict(X_test)
# Evaluate predictions
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))