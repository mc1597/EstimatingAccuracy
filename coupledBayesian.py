import numpy as np
import MySQLdb as db
from scipy.stats import bernoulli
import copy
import csv
import math
import random
import matplotlib.pyplot as plt
import sys
from dp import Dirichlet

def get_bernoulli(p, size):
	rv = bernoulli(p)
	x = np.arange(0, np.minimum(rv.dist.b, 3))
	h = plt.vlines(x, 0, rv.pmf(x), lw=2)
	prb = bernoulli.cdf(x, p)
	h = plt.semilogy(np.abs(x - bernoulli.ppf(prb, p)) + 1e-20)
	labels = bernoulli.rvs(p, size=size).tolist()
	return labels

# def get_errorRates():
# 	errorRates = {}
# 	for function_no in range(numberOfFunctions):
# 		errorRates[function_no] = np.random.beta(errorRatesPriorAlha, errorRatesPriorBeta)
# 	return errorRates

def logBeta(a, b):
	return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

def logSumExp(a, b):
	if math.exp(a) + math.exp(b) <= 0:
		return math.log(0.001)
	else:	
		return math.log(math.exp(a) + math.exp(b))

def get_no_of_instances():
	numberOfInstances = {}
	for domain in domains.values():
		numberOfInstances[domain] = 0
		for instance in sampleInstances:
			if domain in sampleInstances[instance]['labels']:
				numberOfInstances[domain] += 1
	return numberOfInstances

def instance_domain_correspondence():
	in_do_corr = {}
	for p in range(numberOfDomains):
		in_do_corr[p] = []
	for i in range(len(sampleInstances)):
		for j in range(len(sampleInstances[i]['labels'])):
			in_do_corr[sampleInstances[i]['labels'][j]].append(i)
	return in_do_corr

def initialize():
	global labelPriorSamples, errorRateSamples, labelsSamples, disagreements, numberOfInstances, clusterAssignments

	labelPriorSamples = [[None] * numberOfDomains]*numberOfSamples
	errorRateSamples = [[[None] *numberOfFunctions] * numberOfDomains]*numberOfSamples
	labelsSamples = [[None] * numberOfDomains]*numberOfSamples
	disagreements = [[None] * numberOfFunctions]*numberOfDomains
	clusterAssignments = [[None] * numberOfDomains] * numberOfSamples

	for i in range(numberOfSamples):
		for p in range(numberOfDomains):
			labelsSamples[i][p] = [None] * numberOfInstances[p]

	for p in range(numberOfDomains):
		labelPriorSamples[0][p] = 0.5
		
		sum_val = [0] * numberOfInstances[p]
		sum_val1 = [0] * numberOfInstances[p]
		no_of_functions = [0] * numberOfInstances[p]

		for i in range(numberOfInstances[p]):
			for j in range(len(sampleInstances[i]['values'])):
				if sampleInstances[i]['labels'][j] == p:
					sum_val[i] += int(sampleInstances[i]['values'][j] >= 0.5)
					sum_val1[i] += sampleInstances[i]['values'][j]
					no_of_functions[i] += 1


		for i in range(numberOfInstances[p]):
			if sum_val[i] > no_of_functions[i]/2:
				labelsSamples[0][p][i] = 1
			else:
				labelsSamples[0][p][i] = 0

		for j in range(numberOfFunctions):
			errorRateSamples[0][p][j] = 0.25
			disagreements[p][j] = 0

		for i in range(numberOfInstances[p]):
			if p not in sampleInstances[i]['labels']:
				continue
			if no_of_functions[i] > 0:
				mean = int((sum_val1[i] / no_of_functions[i]) >= 0.5)
			else:
				mean = 0

			if mean != labelsSamples[0][p][i]:
				x = samples[i]['labels'].index(p)
				function_id = samples[i]['functions'][x]
				disagreements[p][function_id] += 1
		clusterAssignments[0][p] = p
		dpPrior.addMemberToCluster(clusterAssignments[0][p])
		

def samplePriorAndErrorRates(sample_no):
	for p in range(numberOfDomains):
		labels_count = 0
		for i in range(numberOfInstances[p]):
			labels_count += labelsSamples[sample_no][p][i]
		
		labelPriorSamples[sample_no][p] = np.random.beta(labelsPriorAlpha + labels_count, labelsPriorBeta + numberOfInstances[p] - labels_count)
		disagreements[p] = [0] * numberOfFunctions
		temp_instances_no = [0] * numberOfFunctions
		sum_val = [0] * numberOfInstances[p]
		no_of_functions = [0] * numberOfInstances[p]


		for i in range(numberOfInstances[p]):
			for j in range(len(sampleInstances[i]['values'])):
				if sampleInstances[i]['labels'][j] == p:
					sum_val[i] += sampleInstances[i]['values'][j]
					no_of_functions[i] += 1

		for k in range(numberOfDomains):
			if clusterAssignments[sample_no][k] == p:
				for sample in range(numberOfInstances[p]):
					if p not in sampleInstances[i]['labels']:
						continue
					x = sampleInstances[i]['labels'].index(p)
					function_id = sampleInstances[i]['functions'][x]
					if no_of_functions[i] > 0:
						mean = int((sum_val[i] / no_of_functions[i]) >= 0.5)
					else:
						mean = 0
					
					if labelsSamples[sample_no][p][i] != mean:
						disagreements[p][function_id] += 1
					temp_instances_no[function_id] += 1

		for j in range(numberOfFunctions):
			errorRateSamples[sample_no][p][j] = np.random.beta(errorRatesPriorAlha + disagreements[p][j], errorRatesPriorBeta + temp_instances_no[j] - disagreements[p][j])


def sampleClusterAssignments(sample_no):
	for p in range(numberOfDomains):
		sum_val = [0] * numberOfInstances[p]
		no_of_functions = [0] * numberOfInstances[p]
		for i in range(numberOfInstances[p]):
			for j in range(len(sampleInstances[i]['values'])):
				if sampleInstances[i]['labels'][j] == p:
					sum_val[i] += sampleInstances[i]['values'][j]
					no_of_functions[i] += 1

		dpPrior.removeMemberFromCluster(clusterAssignments[sample_no][p])
		current_no_of_clusters = dpPrior.computeClusterDistribution()
		z_probabilities = [None] * current_no_of_clusters
		for i in range(current_no_of_clusters):
			z_probabilities[i] = math.log(dpPrior.getClusterUnnormalizedProbability(i))
		
		disagreements[p] = [0]* numberOfFunctions
		for i in range(numberOfInstances[p]):
			if p not in sampleInstances[i]['labels']:
				continue
			if no_of_functions[i] > 0:
				mean = int((sum_val[i] / no_of_functions[i]) >= 0.5)
			else:
				mean = 0
			x = sampleInstances[i]['labels'].index(p)
			function_id = sampleInstances[i]['functions'][x]
			if labelsSamples[sample_no][p] != mean:
				disagreements[p][function_id] += 1
		for j in range(numberOfFunctions):
			for i in range(current_no_of_clusters - 1):
				clusterID = dpPrior.getClusterID(i)
				z_probabilities[i] += disagreements[p][j] * math.log(errorRateSamples[sample_no][clusterID][j])
				z_probabilities[i] += (numberOfInstances[p] - disagreements[p][j]) * math.log(1 - errorRateSamples[sample_no][clusterID][j])
			z_probabilities[current_no_of_clusters - 1] += logBeta(errorRatesPriorAlha + disagreements[p][j], errorRatesPriorBeta + numberOfInstances[p] - disagreements[p][j]) - logBeta(errorRatesPriorAlha, errorRatesPriorBeta)

		for i in range(1, current_no_of_clusters):
			z_probabilities[i] = logSumExp(z_probabilities[i - 1], z_probabilities[i])
		uniform = math.log(random.random()) + z_probabilities[current_no_of_clusters - 1]
		new_clusterID = dpPrior.getClusterID(current_no_of_clusters - 1)
		clusterAssignments[sample_no][p] = new_clusterID
		for i in range(current_no_of_clusters - 1):
			if z_probabilities[i] > uniform:
				clusterID = dpPrior.getClusterID(i)
				clusterAssignments[sample_no][p] = clusterID
				dpPrior.addMemberToCluster(clusterID)
				break
		if clusterAssignments[sample_no][p] == new_clusterID:
			dpPrior.addMemberToCluster(new_clusterID) 


def sampleLabels(sample_no):
	for p in range(numberOfDomains):
		sum_val = [0] * numberOfInstances[p]
		no_of_functions = [0] * numberOfInstances[p]
		for i in range(numberOfInstances[p]):
			for j in range(len(sampleInstances[i]['values'])):
				if sampleInstances[i]['labels'][j] == p:
					sum_val[i] += sampleInstances[i]['values'][j]
					no_of_functions[i] += 1
	

		p0 = [1 - labelPriorSamples[sample_no][p]] * numberOfInstances[p]
		p1 = [labelPriorSamples[sample_no][p]] * numberOfInstances[p]
		for i in range(numberOfInstances[p]):
			if p not in sampleInstances[i]['labels']:
				continue
			x = sampleInstances[i]['labels'].index(p)
			function_id = sampleInstances[i]['functions'][x]
			if no_of_functions[i] > 0:
				mean = int((sum_val[i] / no_of_functions[i]) >= 0.5)
			else:
				mean = 0
			if mean == 0:
				p0[i] = 1 - errorRateSamples[sample_no][clusterAssignments[sample_no][p]][function_id]
				p1[i] = errorRateSamples[sample_no][clusterAssignments[sample_no][p]][function_id]
			else:
				p0[i] = errorRateSamples[sample_no][clusterAssignments[sample_no][p]][function_id]
				p1[i] = 1 - errorRateSamples[sample_no][clusterAssignments[sample_no][p]][function_id]
		
		labelsSamples[sample_no][p] = get_bernoulli(p1[i] / (p0[i] + p1[i]), numberOfInstances[p])

def storeSamples(sample_no):
	labelPriorSamples[sample_no] = copy.deepcopy(labelPriorSamples[sample_no - 1])
	errorRateSamples[sample_no] = copy.deepcopy(errorRateSamples[sample_no - 1])
	clusterAssignments[sample_no] = copy.deepcopy(clusterAssignments[sample_no - 1])
	labelsSamples[sample_no] = copy.deepcopy(labelsSamples[sample_no - 1])


def get_inference():
	samplePriorAndErrorRates(0)
	sampleClusterAssignments(0)
	sampleLabels(0)

	for i in range(1, numberOfSamples):
		samplePriorAndErrorRates(i - 1)
		sampleClusterAssignments(i - 1)
		sampleLabels(i - 1)
		storeSamples(i)

	labelMeans = {}
	labelPriorMeans = {}
	errorRateMeans = {}
	for p in range(numberOfDomains):
		labelMeans[p] = [0] * numberOfInstances[p]
		labelPriorMeans[p] = 0
		errorRateMeans[p] = [0] * numberOfFunctions

	for sample_no in range(numberOfSamples):
		for p in range(numberOfDomains):
			labelPriorMeans[p] += labelPriorSamples[sample_no][p]
			for j in range(numberOfFunctions):
				errorRateMeans[p][j] += errorRateSamples[sample_no][clusterAssignments[sample_no][p]][j]
			for i in range(numberOfInstances[p]):
				labelMeans[p][i] += labelsSamples[sample_no][p][i]

	for p in range(numberOfDomains):
		labelPriorMeans[p] /= float(numberOfSamples) # check for float
		for j in range(numberOfFunctions):
			errorRateMeans[p][j] /= float(numberOfSamples) #check for float
		for i in range(numberOfInstances[p]):
			labelMeans[p][i] /= float(numberOfSamples) #check for float

	return labelPriorMeans, errorRateMeans, labelMeans, clusterAssignments


def get_result():
	maxList = [0] * len(sampleInstances)
	maxVal = [0] * len(sampleInstances)
	for p in range(numberOfDomains):
		for i in range(len(in_do_corr[p])):
			if maxVal[in_do_corr[p][i]] <= lmeans[p][in_do_corr[p][i]]:
				maxVal[in_do_corr[p][i]] = lmeans[p][in_do_corr[p][i]]
				maxList[in_do_corr[p][i]] = p
	
	print 'Most likely single label for '
	index = 0
	for i in maxList:
		print 'Instance ' + str(index) + ' is ' + domainsInv[i]
		index += 1

	print '\n'
	print 'Error Rates of Functions with '
	errAvgVal = [0] * numberOfFunctions
	for p in range(numberOfDomains):
		for i in range(len(emeans[p])):
			errAvgVal[i] += emeans[p][i]
	for i in range(len(errAvgVal)):
		errAvgVal[i] = 1 - (errAvgVal[i] / float(numberOfDomains))
		print 'function ID ' + str(i) + ' is ' + str(errAvgVal[i])
	print '\n'
	return errAvgVal, maxList

def get_label_accuracy(estimated_labels, actual_labels):
	miss_classified = 0
	for instance in estimated_labels:
		if estimated_labels[instance] != actual_labels[instance]:
			miss_classified += 1
	return 1 - (miss_classified / float(len(estimated_labels)))


if __name__ == '__main__':
	labelsPriorAlpha = 1.0
	labelsPriorBeta = 1.0
	errorRatesPriorAlha = 1.0
	errorRatesPriorBeta = 2.0
	numberOfFunctions = int(sys.argv[2])
	domainValues = []
	domains = {'animal' : 0, 'city' : 1, 'book' : 2, 'movie' : 3}
	domainsInv = {0 : 'animal', 1 : 'city', 2 : 'book', 3 : 'movie'}
	numberOfDomains = len(domains)
	dpPrior = Dirichlet(1.0, numberOfDomains)
	file_name = sys.argv[1]
	f = open(file_name, 'rb')
	csv_reader = csv.reader(f)
	sampleInstances = {}
	samples = []
	i = 0
	actual_labels = {}
	for row in csv_reader:
		samples.append(row)
		row[0] = int(row[0].strip())
		if row[0] not in actual_labels:
			actual_labels[row[0]] = int(row[4].strip())
		if row[0] in sampleInstances:
			sampleInstances[row[0]]['labels'].append(domains[row[1].strip()])
			sampleInstances[row[0]]['functions'].append(int(row[2].strip()))
			sampleInstances[row[0]]['values'].append(float(row[3].strip()))
		else:
			d = {'labels' : [domains[row[1].strip()]], 'functions' : [int(row[2].strip())], 'values' : [float(row[3].strip())]}
			sampleInstances[row[0]] = d

	numberOfSamples = len(samples)
	#step 1
	priorProbability = np.random.beta(labelsPriorAlpha, labelsPriorBeta)
	
	#step 2
	labels = get_bernoulli(priorProbability, numberOfSamples)

	#step 3
	#errorRates = get_errorRates()	

	#step 4
	numberOfInstances = get_no_of_instances()
	in_do_corr = instance_domain_correspondence()
	initialize()
	plmeans, emeans, lmeans, cluster_assignments = get_inference()
	function_error_Rates, single_most_likely_labels = get_result()
	label_accuracy = get_label_accuracy(single_most_likely_labels, actual_labels)
	print 'Label Accuracy ' + str(label_accuracy)




