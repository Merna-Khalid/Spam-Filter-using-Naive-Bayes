# Spam Filter

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

A Naive bayes based classifier.
The program takes input of filename
then asks which classifier to use (Custom or Sklearn)

Example usage:

	Please enter a filename : spam.csv
	Do you want to use sklearn or custom implementation? (y/n/exit) n
	Metrics for custom implementation : 
	{'Accuracy': 0.9845594913714805, 'Recall': 1.0, 'Precision': 0.896969696969697, 'F1': 0.9456869009584664}
	Please enter a test msg : (msg/exit) : mooooooney come get some money
	This msg is probably Ham.
	Please enter a test msg : (msg/exit) : exit
	Do you want to continue? (y/n/exit) n


