
from random import seed
from random import randrange
from csv import reader
 
# Load a CSV file
def csv(filename):
        data = list()
        with open (filename, 'r') as file:
                csv_reader= reader(file)
                for row in csv_reader:
                        if not row:
                                continue
                        data.append(row)
                return data
	#file = open(filename, 'rb')
	#lines = reader(file)
	#dataset = list(lines)
	#return dataset
 
# Convert string column to float
def str_to_float(data, column):
	for row in data:
		row[column] = float(row[column].strip())
 
# Split a dataset into k folds
def cross_split(dataset, n_folds):
	data_split = list()
	data_copy = list(data)
	fold_size = int(len(data) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(data_copy))
			fold.append(data_copy.pop(index))
		data_split.append(fold)
	return data_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(data, algorithm, n_folds, *args):
	folds = cross_split(data, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores
 
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, data):
	left, right = list(), list()
	for row in data:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
 
# Calculating the Gini index
def gini_index(groups, classes):
	
	n_instances = float(sum([len(group) for group in groups]))
	gini = 0.0
	for group in groups:
		size = float(len(group))
		
		if size == 0:
			continue
		score = 0.0
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		gini += (1.0 - score) * (size / n_instances)
	return gini
 
# Select the best split point for a dataset
def get_split(data):
	class_values = list(set(row[-1] for row in data))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(data[0])-1):
		for row in data:
			groups = test_split(index, row[index], data)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
 

def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)
 

def tree_build(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root
 
# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
 
# Classification and Regression Tree Algorithm
def decision(train, test, max_depth, min_size):
	tree = tree_build(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)
 
seed(1)

filename = '1-diabetes.csv'
data = csv(filename)
for i in range(len(data[0])):
	str_to_float(data, i)
n_folds = 5
max_depth = 5
min_size = 10
scores = evaluate_algorithm(data, decision, n_folds, max_depth, min_size)
#print('Scores: %s' % scores)
print('Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
