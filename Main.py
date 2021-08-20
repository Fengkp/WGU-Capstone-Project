import pandas as pd # provides fast data analysis structures and tools
import numpy as np # faster way of working with arrays
import seaborn as sns # to visualize data
import matplotlib.pyplot as plt
import ipywidgets as widgets
import math
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# read and save .csv file in a 2d structure df
df = pd.read_csv('test_scores.csv')
# identifies the two columns to be worked on
df_binary = df[['pretest', 'posttest']]
# renames columns
df_binary.columns = ['Pre-test Scores', 'Post-test Scores']
print('Data Representation')
df_binary

# visualize the data in a linear regression fit graph
sns.lmplot(x = 'Pre-test Scores', y = 'Post-test Scores', data = df_binary, ci = None, height=7, aspect=1.5)
ax = plt.gca()
ax.set(xlabel = 'Pre-test Scores', ylabel = 'Post-test Scores')
print('Data Visualization 1')
plt.show()

plt.figure(figsize = (12, 7))
sns.histplot(data = df_binary, x = 'Pre-test Scores', color = 'r', label = 'Pre-test Scores')
sns.histplot(data = df_binary, x = 'Post-test Scores', color = 'b', label = 'Post-test Scores')
ax = plt.gca()
ax.set(xlabel = 'Pre-test and Post-test Scores', ylabel = 'Count')
plt.legend() 
print('Data Visualization 2')
plt.show()

plt.figure(figsize=(12,7))
sns.kdeplot(data = df_binary[['Pre-test Scores', 'Post-test Scores']], shade = True)
ax = plt.gca()
ax.set(xlabel = 'Pre-test and Post-test Scores')
print('Data Visualization 3')
plt.show()

# reshapes and separates the columns into x and y arrays
x = np.array(df_binary['Pre-test Scores']).reshape(-1, 1)
y = np.array(df_binary['Post-test Scores']).reshape(-1, 1)

# Train Test Split creates a model using only the training data given, in this case pretest and posttest scores
# and is then evaluated by the test data
X_train, X_test, y_train, y_test = train_test_split(x, y)
regr = LinearRegression()
# creates the best fit line for the model to determine our predictions and how accurate they are
regr.fit(X_train, y_train)

y_predict = regr.predict(X_test)

# this will visualize the training the model
plt.figure(figsize = (12, 7))
plt.scatter(X_train, y_train, label = 'Training Data', color = 'r', alpha = .5)
plt.scatter(X_test, y_test, label = 'Testing Data', color = 'b', alpha = .5)
plt.legend()
print('Projection Visualization 1')
plt.show()
# training data is everything used to make the line for the linear model

# scores the accuracy of the predictions and determines whether this is the appropiate model
print()
print('Score: ', regr.score(X_test, y_test))

# comparing the projections versus the actual results; only looking at first ten
projectionComparison = []
for i in range(10):
    projectionComparison.append([math.floor(X_test[i][0]), math.floor(y_predict[i]), math.floor(y_test[i][0])])

projections = pd.DataFrame(projectionComparison, columns=['Test Score', 'Projected Score', 'Actual Score'])
print('Table with Actual Scores versus Projections')
projections

print('Generate Projection')
input_text = widgets.Text(placeholder = 'Press enter after input...', description = 'Test Score:')
output_text = widgets.Text(description = 'Projection:')

def bind_input_to_output(sender):
    output_text.value = str(math.floor(regr.predict([[input_text.value]])))
    
input_text.on_submit(bind_input_to_output)
display(input_text)
display(output_text)