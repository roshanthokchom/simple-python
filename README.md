"# simple-python" 
"# simple-python" 
"# simple-python" 
from sklearn import tree
import matplotlib.pyplot
%matplotlib inline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
x=cancer.data
y=cancer.target
clf=DecisionTreeClassifier(max_depth=10)
x_train,x_test,y_train,y_test=train_test_split(x,y)
clf=clf.fit(x_train,y_train)
tree.plot_tree(clf.fit(x_train,y_train))
