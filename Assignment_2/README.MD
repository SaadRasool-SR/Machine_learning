# Randomized Search Algorithms

This readme file will address the follwing topics regarding to the code:

1) Env Requirements 
2) Required Code changes 
3) Plots and Results
4) Help
5) Authors

## Env Requirements:

There is two ways to set up the envirnment used for the assignment:
```
1) use the environment.yml file to create local environment by running the following cmd in the terminal: ***conda env create -f environment.yml*** 

2) Mannually download the required libiraries mentioned below:
  - The only libraires used for this project were:  mlrose_hiive, numpy and plotly, sklearn libiraries

## Required Code Changes :
Once the envirnoment is set up, it is important to note that there are four different python script that will be required to run inorder to generate all the plots:

The four scripts are:
1) randomized_opt_Max_color_K.py
2) Randomized_opt_NN.py
3) randomized_opt_queen.py
4) randomized_opt_TSP.py

The biggest and only change that the user will have to make would be to add data path into the code. Please add the path to data on the following lines of the code for each script

1) Randomized_opt_NN.py line(32 and 33)

The data csv files are under the data folder, breast-cancer.csv , heart_test.csv and heart_train.csv, once the repo is cloned user will have to add in the path to the above mention lines. 

Once the path for the data is set, everything else should run as is and generate the required plots

## Help

Any advise for common problems or issues, via creating issues or reachine out


## Authors

Contributors names and contact info
- Saad Rasool
- srasool@ualberta.ca
