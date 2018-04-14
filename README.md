File structure:
1. Two groups of files for two different classification problems (Titanic and Digit). 
2. Each algorithm is saved as a single file. For example: titanic_dt.py contains decision tree classifier implementation and all the tuning process.
3. Naming of each file is: xxx_yyy.py, where xxx stands for classification problem (digit and titanic); and yyy stands for classifier or other functions (i.e. eda)


Supporting files:
1. All files are written and executed in python 3.6.4
2. Supporting packages include:
	1) Jinja2 2.10 (for eda ploting, not related to algorithms)
	2) Keras 2.1.3
	3) libgcc-ng 7.2.0
	4) matplotlib 2.1.2
	5) numpy 1.14.0
	6) pandas 0.22.0
	7) scikit-learn 0.19.1
	8) scipy 1.0.0
	9) seaborn 0.8.1
	10) tensorflow 1.5.0
	11) tensorflow-tensorboard 1.5.0


Run:
To see the results of each algorithm, just run the corresponding .py file. 
example: to see Neural Network for digit problem --> run digit_NN.py
* dataset file for digit problem: sampled_digit.csv
* dataset file for titanic problem: titanic_data.csv
* dataset, and corresponding helpers.py file and config.py file need to be put in the same folder together with algorithm files.


