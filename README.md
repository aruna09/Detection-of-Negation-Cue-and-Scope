# Detection-of-Negation-Scope-and-Cue

This is a problem statement of the Philips Datascience Hackathon 2018 held at the Philips Innovation Campus(PIC), Banglore. The top 15 finalist teams were expected to give a presentation of their approach on the same.

## Why perform negation cue and scope detection?
In clinical data, negation detection plays a major role for understanding the status of patients health and understanding of what doctor wants to say. Having said that there's been research going on to find the best solution in terms of closely predicting the scope.

## What is negation cue?
The words which change the polarity of the sentence and make them negative are called negation cues. For eg take the sentence "I don't like you". In this sentence "don't" is a negation cue.

## What is a negation scope?
The part of the sentence which gets effected by the negation cue is called the scope of that particular cue. 

## What we have used?
We have used the SVM classifier for negation cue and scope detection.

## Why did we not use neural nets for the same?
The first thought that might strike anyone to solve a problem of this nature is to use a neural net(a bi-LSTM probably). However if you closely observe the dataset, you would notice that it is very sparse. A neuralnet would surely overfit, which it indeed did when we tried to use the same. Hence SVM was the best alternative for the given constraint.

## Challenges faced
1. Overlapping scopes possible
2. Ambigous words like, "infrequent"(negation cue) and "inline"(not a negation cue") encountered.
