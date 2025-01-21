import pandas as pd
import numpy as np

# Load the data from the Excel file
file_path = r'G:\Mechin learning lab\example.xlsx'
data = pd.read_excel(file_path)

# Function to implement Find-S algorithm
def find_s_algorithm(data):
    # Extract the feature columns and the target column
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Initialize the hypothesis
    hypothesis = ['0'] * len(X[0])
    
    # Iterate through each instance
    for i in range(len(X)):
        # Only consider positive instances
        if y[i] == 'Yes':
            for j in range(len(hypothesis)):
                if hypothesis[j] == '0':
                    hypothesis[j] = X[i][j]
                elif hypothesis[j] != X[i][j]:
                    hypothesis[j] = '?'
    
    return hypothesis

# Apply the Find-S algorithm to the data
hypothesis = find_s_algorithm(data)
print('Final Hypothesis:', hypothesis)
