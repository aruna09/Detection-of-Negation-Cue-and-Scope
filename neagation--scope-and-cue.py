from detect_scopes2 import *
import networkx as nx
import spacy

def training_phase(filename):
	cue_ground_truths = []
	fp = open(filename, 'r')
	data = fp.readlines()
	corpus = []
	for i in data:
		i = i.replace('\n', '')
		corpus.append(i.split('\t'))

	postag = []

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
	
	wc_list = []

	temp_corpus = corpus
	temp_cue_ground_truths = cue_ground_truths
	for i in sent_index:
		i = int(i)
		target_sentence = temp_corpus[:i+1]
		target_cues = temp_cue_ground_truths[:i+1]
		temp_corpus = temp_corpus[i+1:]
		temp_cue_ground_truths = temp_cue_ground_truths[i+1:]
		for j in range(len(target_cues)):
			if (target_cues[j] == 1):
				a = []
				for k in range(len(target_cues)):
					a.append(target_sentence[j][3])
					a.append(target_sentence[k][3])
					
				wc_list.append(a)
				a = []	
	# print(wc_list[1])
	# wc_list = np.array(wc_list)
	# print(wc_list.shape)
	sentences = []
	cues_count = []
	temp_corpus = corpus
	temp_cue_ground_truths = cue_ground_truths
	for i in sent_index:
		i = int(i)
		c = 0
		target_sentence = temp_corpus[:i+1]
		target_cues = temp_cue_ground_truths[:i+1]
		temp_corpus = temp_corpus[i+1:]
		temp_cue_ground_truths = temp_cue_ground_truths[i+1:]
		for j in range(len(target_cues)):
			if(target_cues[j] == 1):
				c += 1
		for j in range(len(target_cues)):
			if (target_cues[j] == 1):
				a = []
				for k in range(len(target_cues)):
					a.append(target_sentence[k][3])
					
				s = " ".join(a)
				sentences.append(s)
				cues_count.append(c)
				a = []	
				break
	# print(sentences[1])
	# print(cues_count[1])

	GloveEmbeddings = {}
	fe = open("glove.6B.50d.txt","r",encoding="utf-8",errors="ignore")
	for line in fe:
	    tokens= line.strip().split()
	    word = tokens[0]
	    vec = []
	    la = tokens[1:]
	    for i in range(len(la)):
	    	sa = float(la[i])
	    	vec.append(sa)
	    GloveEmbeddings[word]=vec
	sodi = []
	for i in range(50):
		sodi.append(0.0)
	GloveEmbeddings["zerovec"] = sodi
	fe.close()

	sl_paths = []
	X_train = []
	c = 0
	print(len(sentences))
	for sentence in sentences:
		b = []
		modified_sentence = nlp(str(sentence))
		# print(modified_sentence)
		g = create_graph(modified_sentence)
		for i in cues_count:
			i = int(i)
			for k in range(i):
				for j in range(0,len(wc_list[c+k])-1,2):
					source = wc_list[c+k][j]
					target = wc_list[c+k][j+1]
					if (g.has_node(source) and g.has_node(target)):
						# print(modified_sentence)
						try:
							# print(source)
							# print(target)
							path_length = nx.shortest_path_length(g, source, target)
							b.append(path_length)
						except nx.NetworkXNoPath:
							b.append(0)
							# print(modified_sentence)
							# print(source)
							# print(target)
					else:
						b.append(-1)
					if(source in GloveEmbeddings):
						glove_source = GloveEmbeddings[source] 
					else:
						glove_source = GloveEmbeddings["zerovec"]
					if(target in GloveEmbeddings):
						glove_target = GloveEmbeddings[target]
					else:
						glove_target = GloveEmbeddings["zerovec"]
					if(str(path_length) in GloveEmbeddings):
						glove_path_length = GloveEmbeddings[str(int(path_length))]
					else:
						glove_path_length = GloveEmbeddings["zerovec"]
					line = [glove_source, glove_target, glove_path_length]
					X_train.append(line)
				sl_paths.append(b)
				b = []
		b = []
		c += 1

	# print(sl_paths[1])
	# X_train_1 = np.array(X_train[:int(len(X_train)/2)])
	# X_train_2 = np.array(X_train[int(len(X_train)/2):])
	# X_train = np.vstack(X_train_1, X_train_2)
	# print(X_train.shape)
	# print(X_train[3])
	y_train = []
	temp_corpus = corpus
	temp_cue_ground_truths = cue_ground_truths
	features = []
	ground_scope = []
	for i in sent_index:
		i = int(i)
		target_sentence = temp_corpus[:i+1]
		target_cues = temp_cue_ground_truths[:i+1]
		temp_corpus = temp_corpus[i+1:]
		temp_cue_ground_truths = temp_cue_ground_truths[i+1:]
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
						y_train.append(1.0)
					else:
						y_train.append(0.0)

	# print(y_train[:32])
	print(len(X_train))
	test = np.array(X_train[0])
	print(test.shape)
	# temp = convert_to_numpy(X_train)
	# X_train, y_train, X_validate, y_validate = train_test_split(temp, y_train, test_size = 0.2, random_state = 101)
	# svm = SVC(kernel="linear", C=0.0025, random_state = 101)
	# svm.fit(X_train, y_train)

	# y_predict = svm.predict(X_validate)
	# f1_measure = f1_score(y_validate, y_predict, average="macro")
	# print("Training phase f1_score is " + str(f1_measure))
	# pickle.dump(svm, open("scope_detector.sav", 'wb'))

def convert_to_numpy(X_train):
	temp = []
	for i in range(len(X_train)):
		t = np.array(X_train[i])
		row, col = t.shape
		t1 = t.reshape(row*col)
		temp.append(t1)
	return temp

def create_graph(sentence, e1=None, e2=None):
    # print("This is create graph")
    edges = []
    for token in sentence:
        for child in token.children:
            edges.append(('{0}'.format(token),
                          '{0}'.format(child)))
    graph = nx.Graph(edges)
    return graph

if __name__ == '__main__':
	nlp = spacy.load('en_core_web_sm')
	training_phase("corpus/training/SEM-2012-SharedTask-CD-SCO-training-09032012.txt")