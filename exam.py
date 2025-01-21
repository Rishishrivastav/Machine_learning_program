import pandas as pd


file_path = r'G:\Mechin learning lab\exam.xlsx'
data = pd.read_excel(file_path)


def find_s_algorithm(data):
    target_column = data.columns[-1]

    
    X = data.iloc[:, :-1].values  
    y = data[target_column].values  
    
    
    num_features = len(X[0])
    hypothesis = ['0'] * num_features
    
    
    for i in range(len(X)):
        
        if y[i] == 'Yes':
            for j in range(num_features):
                if hypothesis[j] == '0':
                    hypothesis[j] = X[i][j]
                elif hypothesis[j] != X[i][j]:
                    hypothesis[j] = '?'
    
    return hypothesis


hypothesis = find_s_algorithm(data)
print('Final Hypothesis:', hypothesis)
