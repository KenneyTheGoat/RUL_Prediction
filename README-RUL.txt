ALL PROGRAMS ARE MEANT TO READ FILES IN THE SAME WORKING DIRECTORY
OTHERWISE MANUALLY SPECIFY THE FILEPATH IF READING FROM A DIFFERENT DIRECTORY
--------------------------------------------------------------------------------------------------------------------------
Dataset.py

To run this program from the current working directory.
1. Open the terminal
2. type 'python Dataset.py' and press enter
3. enter battery ID

This will read the NASA .mat file and converts it into a .csv

--------------------------------------------------------------------------------------------------------------------------
Merge.py

To run this program from the current working directory.
1. Open the terminal
2. type 'python Merge.py' and press enter

This will merge all the specified NASA .csv files into one data frame

--------------------------------------------------------------------------------------------------------------------------
Balanced.py

To run this program from the current working directory.
1. Open the terminal
2. type 'python Balanced.py' and press enter

This is meant to take in merged NASA dataset for different batteries, clean it and balance the number of data points for different batteries to be equal to the battery type that has the least number of points. Meant to be used for EDA, training and testing.

--------------------------------------------------------------------------------------------------------------------------
NNetwork.py

To run this program from the current working directory.
1. Make sure line 21 is correct and reads in the balanced and cleaned merged dataset for training.
2. Open the terminal
3. type 'NNetwork.py' and press enter

This will start the training of the 1D Convolutional Neural Network on the data and will save the trained model to the same working directory as the model definition.

-------------------------------------------------------------------------------------------------------------------------- 
RForest.py

To run this program from the current working directory.
1. Make sure line 20 is correct and reads in the balanced and cleaned merged dataset for training.
2. Open the terminal
3. type 'RForest.py' and press enter

This will start the training of the Random Forest classifier on the data and will save the trained model to the same working directory as the model definition.

--------------------------------------------------------------------------------------------------------------------------
Predict.py

To run this program from the current working directory.
1. Open the terminal
2. type 'python Predict.py --csv name_of_file.csv --model cnn/rf/ct' and press enter
	where name_of_file.csv is the name of the single NASA battery data file you want to make a classification for 	using its features where cnn=argument for using the C. Neural Network, rf=Random forest, ct=Classification tree.

This will make a classification of the battery type for the specified battery dataset with the specified model to use.

--------------------------------------------------------------------------------------------------------------------------
Regression.py

To run this program from the current working directory.
1. Specify parameters in line 117.
2. type 'python Regression.py' and press enter

This will load a trained regression model or use the training data specified in line 105.
If you want to train a new model with new data, specify the filename in 105 and run again.

--------------------------------------------------------------------------------------------------------------------------
CTree.R, EDA.R, XG-MLR.R

To run these scripts from the current working directory.

1.Highlight all the code and press Ctrl+Enter

To run a single line of code.

1. Highlight the line of code and press Ctrl+Enter

Make sure you check for dependencies before running a line of code, if not running the entire script at once

--------------------------------------------------------------------------------------------------------------------------



