import numpy as np
import nltk
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.problem_transform import BinaryRelevance
# import warnings
# warnings.filterwarnings('always')
import sys
import os
import argparse
import pickle
from collections import Counter
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional, Activation, Dropout
from keras.layers import Flatten
from keras.activations import softmax

def training_phase(filename):
	cue_ground_truths = []
	fp = open(filename, 'r')
	data = fp.readlines()
	corpus = []
	for i in data:
		i = i.replace('\n', '')
		corpus.append(i.split('\t'))

	postag = []
	# feature extraction
	for line in range(len(corpus)):
		if (len(corpus[line])>8):
			count = int((len(corpus[line])-7)/3)
			pakodi = []
			for j in range(count):
				word = corpus[line][7+(j*3)]
				pakodi.append(word)
			if(corpus[line][3] in pakodi):
				cue_ground_truths.append(1)
				postag.append(corpus[line][5].lower())	
				
			else:
				cue_ground_truths.append(0)
				postag.append(corpus[line][5].lower())	

		elif (len(corpus[line]) == 8):
			cue_ground_truths.append(0)
			postag.append(corpus[line][5].lower())	

	pos_tags = []
	for i in postag:
		if i not in pos_tags:
			pos_tags.append(i)

	# one-hot-encoding the postags
	one_hot_postag = [] 

	for i in range(len(postag)):
		seq = []
		if(postag[i] in pos_tags):
			req = pos_tags.index(postag[i])
			for j in range(len(pos_tags)):
				if (j == req):
					seq.append(1.0)
				else:
					seq.append(0.0)
		remaining = 100-len(pos_tags)
		for j in range(remaining):
			seq.append(0.0)
		one_hot_postag.append(seq)

	zero_list = []
	for i in range(100):
		zero_list.append(0.0)

	sent_index = []
	for i in range(len(corpus)):
		if(len(corpus[i]) == 1):
			sent_index.append(corpus[i-1][2])
	
	temp = corpus
	corpus = []
	for i in range(len(temp)):
		if(len(temp[i]) == 1):
			continue
		else:
			corpus.append(temp[i])
	
	# uncomment the below lines to take only before 5 and next 6 postags as features
	# temp_corpus = corpus
	# temp_cue_ground_truths = cue_ground_truths
	# temp_one_hot_postag = one_hot_postag
	# features = []
	# cue_postag_features = []
	# for i in sent_index:
	# 	i = int(i)
	# 	target_sentence = temp_corpus[:i+1]
	# 	target_cues = temp_cue_ground_truths[:i+1]
	# 	target_pos = temp_one_hot_postag[:i+1]
	# 	temp_corpus = temp_corpus[i+1:]
	# 	temp_cue_ground_truths = temp_cue_ground_truths[i+1:]
	# 	temp_one_hot_postag = temp_one_hot_postag[i+1:]
	# 	for j in range(len(target_cues)):
	# 		if (target_cues[j] == 1):
	# 			missing = 6 - j
	# 			if(missing>0):
	# 				for k in range(missing):
	# 					features.append(zero_list)
	# 				n = 0
	# 				while n<=j:
	# 					features.append(target_pos[n])
	# 					n = n + 1
	# 			else:
	# 				n = j-6
	# 				while n <= j:
	# 					features.append(target_pos[n])
	# 					n = n +1
	# 			missing = 7 - len(target_sentence) + j 
	# 			if missing>0:
	# 				n = j + 1
	# 				while n < len(target_sentence):
	# 					features.append(target_pos[n])
	# 					n = n+1
	# 				for n in range(missing):
	# 					features.append(zero_list)
	# 			else:
	# 				n = j+1
	# 				while n < (j+7):
	# 					features.append(target_pos[n])
	# 					n = n+1
	# 			cue_postag_features.append(features)
	# 			features = []

	# for i in range(len(cue_postag_features)):
	# 	if(len(cue_postag_features[i]) != 13):
	# 		print(len(cue_postag_features[i]), end= " ")
	# 		print(i)
	# for j in range(len(cue_postag_features[0])):
	# 	for k in range(len(cue_postag_features[0][j])):
	# 		if cue_postag_features[0][j][k] == 1:
	# 			print(pos_tags[k], end= " ")

	temp_corpus = corpus
	temp_cue_ground_truths = cue_ground_truths
	temp_one_hot_postag = one_hot_postag
	features = []
	feature1 = []
	cue_postag_features = []
	for i in sent_index:
		i = int(i)
		target_sentence = temp_corpus[:i+1]
		target_cues = temp_cue_ground_truths[:i+1]
		target_pos = temp_one_hot_postag[:i+1]
		temp_corpus = temp_corpus[i+1:]
		temp_cue_ground_truths = temp_cue_ground_truths[i+1:]
		temp_one_hot_postag = temp_one_hot_postag[i+1:]
		for j in range(len(target_cues)):
			if (target_cues[j] == 1):
				for k in range(len(target_pos)):
					features.append(target_pos[j])
					for l in range(100):
						if l==j:
							feature1.append(1.0)
						else:
							feature1.append(0.0)
					features.append(feature1)
					feature1 = []
				cue_postag_features.append(features)
				features = []	

	cue_postag_features = keras.preprocessing.sequence.pad_sequences(cue_postag_features, maxlen=100)

	cue_postag_features = np.array(cue_postag_features)
	print(cue_postag_features.shape)

	temp_corpus = corpus
	temp_cue_ground_truths = cue_ground_truths
	temp_one_hot_postag = one_hot_postag
	features = []
	ground_scope = []
	for i in sent_index:
		i = int(i)
		target_sentence = temp_corpus[:i+1]
		target_cues = temp_cue_ground_truths[:i+1]
		target_pos = temp_one_hot_postag[:i+1]
		temp_corpus = temp_corpus[i+1:]
		temp_cue_ground_truths = temp_cue_ground_truths[i+1:]
		temp_one_hot_postag = temp_one_hot_postag[i+1:]
		for j in range(len(target_cues)):
			if (target_cues[j] == 1):
				cue_count = int((len(target_sentence[j]) - 7)/3)
				paks = []
				for k in range(cue_count):
					word = target_sentence[j][7+(k*3)]
					paks.append(word)
				indi = 0
				for k in range(cue_count):
					if(paks[k] != '_'):
						indi = k
				for k in range(len(target_sentence)):
					thing = target_sentence[k][7+(indi*3)+1]
					if(thing != '_'):
						features.append(1.0)
					else:
						features.append(0.0)
				ground_scope.append(features)
				features = []
	
	ground_scope = keras.preprocessing.sequence.pad_sequences(ground_scope, maxlen=100)
	ground_scope = np.array(ground_scope)
	print(ground_scope.shape)
	
	# X_train = cue_postag_features
	# y_train = ground_scope

	# nsamples, nx, ny = X_train.shape
	# X_train_2d_bef = X_train.reshape((nsamples,nx*ny))

	# y_train = y_train.reshape(nsamples*nx)

	# svm = SVC(kernel="linear", C=0.0025, random_state = 101)
	# svm.fit(X_train_2d_bef, y_train)
	# pickle.dump(svm, open("scope_detector.sav", 'wb'))

	

	X_train = cue_postag_features
	# y_train = ground_scope

	nsamples, col, vec = X_train.shape
	
	X_train_2d_bef = X_train.reshape((nsamples,col*vec))
	y_train_2d_bef = np.array(ground_scope, dtype = float)
	
	X_train_2d, X_validate_2d, y_train, y_validate = train_test_split(X_train_2d_bef, y_train_2d_bef , test_size = 0.4, random_state = 101)
	
	
	# classifier = LabelPowerset(GaussianNB())#0.49
	# classifier = LabelPowerset(RandomForestClassifier(n_estimators=25)).577
	classifier = LabelPowerset(RandomForestClassifier(n_estimators=50))#.586
	# classifier = BinaryRelevance(MLPClassifier(hidden_layer_sizes = 200, verbose=True, max_iter = 15,learning_rate_init = 0.0035 ))
	# classifier = LabelPowerset(MLPClassifier()) # 0.58
	# classifier = LabelPowerset(MLPClassifier(hidden_layer_sizes = 200, verbose=True)) #0.589
	# classifier = LabelPowerset(MLPClassifier(hidden_layer_sizes = 200, verbose=True, max_iter = 400)) #.584 
	# classifier = LabelPowerset(MLPClassifier(hidden_layer_sizes = 400, verbose=True, learning_rate_init = 0.0035))
	# classifier = LabelPowerset(GaussianNB())0.18
	# classifier = ClassifierChain(DecisionTreeClassifier()) 0.17
	# classifier = ClassifierChain(GaussianNB())0.02
	# classifier = ClassifierChain(GaussianNB())#0.2
	# classifier = BinaryRelevance(GaussianNB()).35 withoutcue
	# classifier = BinaryRelevance(DecisionTreeClassifier())0.15
	# classifier = BinaryRelevance(OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1))0.41
	# classifier = BinaryRelevance(OneClassSVM(nu=0.5, kernel="rbf", gamma=0.5))
	# classifier = BinaryRelevance(RandomForestClassifier(n_estimators=25))0.29
	# classifier = BinaryRelevance(RandomForestClassifier(n_estimators=100))0.24
	# classifier = BinaryRelevance(RandomForestClassifier(n_estimators=3)).2900
	# classifier = BinaryRelevance(MLPClassifier()).22
	classifier.fit(X_train_2d,y_train )
	y_predict = classifier.predict(X_validate_2d)	
	f1_measure = f1_score(y_validate, y_predict, average="weighted",)
	print(f1_measure)
	pickle.dump(classifier, open("scope_detector.sav", 'wb'))








	# X_train = cue_postag_features
	# y_train = ground_scope

	# nsamples, col, vec = X_train.shape

	# model = Sequential()
	# model.add(Bidirectional(LSTM(200, return_sequences=True), input_shape=(col,vec)))
	# # model.add(Bidirectional(LSTM(200, return_sequences=True)))
	# model.add(Dropout(0.2))
	# # model.add(Dense(300))
	# model.add(TimeDistributed(Dense(500)))
	# model.add(Flatten())
	# model.add(Dense(int(col/2), activation = 'relu'))
	# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
	# model.fit(X_train,y_train,epochs = 30, batch_size=100, shuffle=True,verbose=1)
	# model.save_weights('scope-200.h5')

def testing_phase(filename):
	cue_ground_truths = []
	fp = open(filename, 'r')
	data = fp.readlines()
	corpus = []
	for i in data:
		i = i.replace('\n', '')
		corpus.append(i.split('\t'))

	postag = []
	# feature extraction
	for line in range(len(corpus)):
		if (len(corpus[line])>8):
			count = int((len(corpus[line])-7)/3)
			pakodi = []
			for j in range(count):
				word = corpus[line][7+(j*3)]
				pakodi.append(word)
			if(corpus[line][3] in pakodi):
				cue_ground_truths.append(1)
				postag.append(corpus[line][5].lower())	
				
			else:
				cue_ground_truths.append(0)
				postag.append(corpus[line][5].lower())	

		elif (len(corpus[line]) == 8):
			cue_ground_truths.append(0)
			postag.append(corpus[line][5].lower())	

	pos_tags = []
	for i in postag:
		if i not in pos_tags:
			pos_tags.append(i)

	# one-hot-encoding the postags
	one_hot_postag = [] 

	for i in range(len(postag)):
		seq = []
		if(postag[i] in pos_tags):
			req = pos_tags.index(postag[i])
			for j in range(len(pos_tags)):
				if (j == req):
					seq.append(1.0)
				else:
					seq.append(0.0)
		remaining = 100-len(pos_tags)
		for j in range(remaining):
			seq.append(0.0)
		one_hot_postag.append(seq)

	zero_list = []
	for i in range(100):
		zero_list.append(0.0)

	sent_index = []
	for i in range(len(corpus)):
		if(len(corpus[i]) == 1):
			sent_index.append(corpus[i-1][2])
	
	temp = corpus
	corpus = []
	for i in range(len(temp)):
		if(len(temp[i]) == 1):
			continue
		else:
			corpus.append(temp[i])
	
	# uncomment the below lines to take only before 5 and next 6 postags as features
	# temp_corpus = corpus
	# temp_cue_ground_truths = cue_ground_truths
	# temp_one_hot_postag = one_hot_postag
	# features = []
	# cue_postag_features = []
	# for i in sent_index:
	# 	i = int(i)
	# 	target_sentence = temp_corpus[:i+1]
	# 	target_cues = temp_cue_ground_truths[:i+1]
	# 	target_pos = temp_one_hot_postag[:i+1]
	# 	temp_corpus = temp_corpus[i+1:]
	# 	temp_cue_ground_truths = temp_cue_ground_truths[i+1:]
	# 	temp_one_hot_postag = temp_one_hot_postag[i+1:]
	# 	for j in range(len(target_cues)):
	# 		if (target_cues[j] == 1):
	# 			missing = 6 - j
	# 			if(missing>0):
	# 				for k in range(missing):
	# 					features.append(zero_list)
	# 				n = 0
	# 				while n<=j:
	# 					features.append(target_pos[n])
	# 					n = n + 1
	# 			else:
	# 				n = j-6
	# 				while n <= j:
	# 					features.append(target_pos[n])
	# 					n = n +1
	# 			missing = 7 - len(target_sentence) + j 
	# 			if missing>0:
	# 				n = j + 1
	# 				while n < len(target_sentence):
	# 					features.append(target_pos[n])
	# 					n = n+1
	# 				for n in range(missing):
	# 					features.append(zero_list)
	# 			else:
	# 				n = j+1
	# 				while n < (j+7):
	# 					features.append(target_pos[n])
	# 					n = n+1
	# 			cue_postag_features.append(features)
	# 			features = []

	# for i in range(len(cue_postag_features)):
	# 	if(len(cue_postag_features[i]) != 13):
	# 		print(len(cue_postag_features[i]), end= " ")
	# 		print(i)
	# for j in range(len(cue_postag_features[0])):
	# 	for k in range(len(cue_postag_features[0][j])):
	# 		if cue_postag_features[0][j][k] == 1:
	# 			print(pos_tags[k], end= " ")

	temp_corpus = corpus
	temp_cue_ground_truths = cue_ground_truths
	temp_one_hot_postag = one_hot_postag
	features = []
	feature1 = []
	cue_postag_features = []
	for i in sent_index:
		i = int(i)
		target_sentence = temp_corpus[:i+1]
		target_cues = temp_cue_ground_truths[:i+1]
		target_pos = temp_one_hot_postag[:i+1]
		temp_corpus = temp_corpus[i+1:]
		temp_cue_ground_truths = temp_cue_ground_truths[i+1:]
		temp_one_hot_postag = temp_one_hot_postag[i+1:]
		for j in range(len(target_cues)):
			if (target_cues[j] == 1):
				for k in range(len(target_pos)):
					features.append(target_pos[j])
					for l in range(100):
						if l==j:
							feature1.append(1.0)
						else:
							feature1.append(0.0)
					features.append(feature1)
					feature1 = []
				cue_postag_features.append(features)
				features = []	

	cue_postag_features = keras.preprocessing.sequence.pad_sequences(cue_postag_features, maxlen=100)

	cue_postag_features = np.array(cue_postag_features)
	print(cue_postag_features.shape)

	temp_corpus = corpus
	temp_cue_ground_truths = cue_ground_truths
	temp_one_hot_postag = one_hot_postag
	features = []
	ground_scope = []
	for i in sent_index:
		i = int(i)
		target_sentence = temp_corpus[:i+1]
		target_cues = temp_cue_ground_truths[:i+1]
		target_pos = temp_one_hot_postag[:i+1]
		temp_corpus = temp_corpus[i+1:]
		temp_cue_ground_truths = temp_cue_ground_truths[i+1:]
		temp_one_hot_postag = temp_one_hot_postag[i+1:]
		for j in range(len(target_cues)):
			if (target_cues[j] == 1):
				cue_count = int((len(target_sentence[j]) - 7)/3)
				paks = []
				for k in range(cue_count):
					word = target_sentence[j][7+(k*3)]
					paks.append(word)
				indi = 0
				for k in range(cue_count):
					if(paks[k] != '_'):
						indi = k
				for k in range(len(target_sentence)):
					thing = target_sentence[k][7+(indi*3)+1]
					if(thing != '_'):
						features.append(1.0)
					else:
						features.append(0.0)
				ground_scope.append(features)
				features = []
	

	ground_scope = keras.preprocessing.sequence.pad_sequences(ground_scope, maxlen=100)
	ground_scope = np.array(ground_scope)
	print(ground_scope.shape)

	nsamples , col, vec = cue_postag_features.shape
	
	X = cue_postag_features
	Y = ground_scope


	# classifier = LabelPowerset(GaussianNB())0.18
	# classifier = LabelPowerset(GaussianNB())0.56
	# classifier = ClassifierChain(DecisionTreeClassifier()) 0.17
	# classifier = ClassifierChain(GaussianNB())0.02
	# classifier = ClassifierChain(GaussianNB())
	# classifier = BinaryRelevance(GaussianNB()).36
	# classifier = BinaryRelevance(DecisionTreeClassifier())0.15
	# classifier = BinaryRelevance(OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1))
	# classifier = BinaryRelevance(OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1))0.41 withoutcue
	# classifier = BinaryRelevance(OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1))
	# classifier = BinaryRelevance(RandomForestClassifier(n_estimators=25))0.29
	# classifier = BinaryRelevance(RandomForestClassifier(n_estimators=100))0.24
	# classifier = BinaryRelevance(RandomForestClassifier(n_estimators=3)).2900
	# classifier = BinaryRelevance(MLPClassifier()).22
		
	X_train_2d = X.reshape((nsamples,col*vec))

	load_model = pickle.load(open("scope_detector.sav",'rb'))
	y_predict = load_model.predict(X_train_2d)
	a=f1_score(ground_scope,y_predict,average='macro')
	print(a)




	# model = Sequential()
	# model.add(Bidirectional(LSTM(200, return_sequences=True), input_shape=(col,vec)))
	# # model.add(Bidirectional(LSTM(200, return_sequences=True)))
	# model.add(Dropout(0.2))
	# # model.add(Dense(300))
	# model.add(TimeDistributed(Dense(500)))
	# model.add(Flatten())
	# model.add(Dense(int(col/2), activation = 'relu'))
	# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


	# model.load_weights('scope-200.h5')

	# y_pred = model.predict(X)
	# direct = y_pred
	
	# for i in range(len(y_pred)):
	# 	for j in range(len(y_pred[i])):
	# 		if(y_pred[i][j] != 0):
	# 			y_pred[i][j] = 1.0
	# 		else:
	# 			y_pred[i][j] = 0.0
	# # print(y_pred[0])

	# ground_scope = np.array(ground_scope, dtype = float)
	# print((ground_scope[0]))
	# print(len(y_pred[0]))
	# print(len(ground_scope[0]))
	# f1_measure = f1_score(ground_scope, y_pred, average="macro")
	# print(f1_measure)
	# print(direct[1])
	# for i in range(len(ground_scope[0])):
	# 	print(str(ground_scope[1][i]) + " " + str(y_pred[1][i]))


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--path", required = True, help = "path to the training file")
	ap.add_argument("-m", "--mode", required = True, help = "training or testing?")
	args = vars(ap.parse_args())
	file_path = args["path"]
	mode = args["mode"]
	if(mode == "training"):
		training_phase(file_path)
	elif(mode == "testing"):
		testing_phase(file_path)
