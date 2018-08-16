import csv
import random
import math

def Csv(filename):
	line = csv.reader(open(filename, "r"))
	data = list(line)
	for i in range(len(data)):
		data[i] = [float(x) for x in data[i]]
	return data

def split(data, split_ratio):
	train_size = int(len(data) * split_ratio)
	train_set = []
	copy = list(data)
	while len(train_set) < train_size:
		index = random.randrange(len(copy))
		train_set.append(copy.pop(index))
	return [train_set, copy]

def separate(data):
	separated = {}
	for i in range(len(data)):
		vector = data[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(n):
	return sum(n)/float(len(n))

def stdev(n):
	avg = mean(n)
	variance = sum([pow(x-avg,2) for x in n])/float(len(n)-1)
	return math.sqrt(variance)

def summarize(data):
	summary = [(mean(attribute), stdev(attribute)) for attribute in zip(*data)]
	del summary[-1]
	return summary

def summarize_class(data):
	separated = separate(data)
	summary = {}
	for class_value, instance in separated.items():
		summary[class_value] = summarize(instance)
	return summary

def Probability(x, mean, stdev):
	exp = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exp

def class_probability(summary, input_vector):
	probability= {}
	for class_value, class_summary in summary.items():
		probability[class_value] = 1
		for i in range(len(class_summary)):
			mean, stdev = class_summary[i]
			x = input_vector[i]
			probability[class_value] *= Probability(x, mean, stdev)
	return probability
			
def predict(summary, input_vector):
	probability = class_probability(summary, input_vector)
	label, prob = None, -1
	for class_value, predict_prob in probability.items():
		if label is None or predict_prob > prob:
			prob = predict_prob
			label = class_value
	return label

def Prediction(summary, test_set):
	p = []
	for i in range(len(test_set)):
		result = predict(summary, test_set[i])
		p.append(result)
	return p

def Accuracy(test_set, p):
	correct = 0
	for i in range(len(test_set)):
		if test_set[i][-1] == p[i]:
			correct += 1
	return (correct/float(len(test_set))) * 100.0

def main():
	filename = '1-diabetes.csv'
	split_ratio = 0.69
	data = Csv(filename)
	training_set, test_set = split(data, split_ratio)
	print(('Split {0} rows into train={1} and test={2} rows').format(len(data), len(training_set), len(test_set)))
	summary = summarize_class(training_set)
	p = Prediction(summary, test_set)
	accuracy = Accuracy(test_set, p)
	print(('Accuracy: {0}%').format(accuracy))

main()
