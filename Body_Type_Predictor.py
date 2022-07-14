# Carter McKinnon Machine Learning Capstone Codecademy
# All data has been provided by codecademy
# Goal: Use KNN to predict body type based on several factors
# All research, rough coding, and the sentiment learning models are in jupyter notebook
# I attempted to use a sentiment learning model to add another factor to the data but the change was very minimal so I decided not to use it

# Importing relevant packages and loading the profile data into a dataframe
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class format_data:
    def __init__ (self, dataframe):
        self.dataframe = dataframe
    
    # function that removes all the rows of a dataframe where the target column value in that row is equal to nan
    def remove_rows(self, target_column):
        # creating a list of not null values
        null_values = self.dataframe[target_column].isna()
        rows_to_keep = [i for i in range(len(null_values)) if not null_values[i]]
        print('removing rows...')
        # creating new dataframe that does not contain rows where body type was n/a
        cleaned_data = self.dataframe[self.dataframe.index.isin(rows_to_keep)]
        print('rows removed!')
        self.dataframe = cleaned_data
        return cleaned_data

    # function that removes NaN values, replaces column values based on array of dictionary values, and scales data
    def remove_nan_replace (self, column_arr, replacement_values_arr):
        for column in column_arr:
            for i in range(len(replacement_values_arr)):
                self.dataframe[column].replace(replacement_values_arr[i], inplace=True)
            
            self.dataframe.dropna(axis=0, how='any', inplace=True)
        formatted_data = self.dataframe
        return formatted_data
    
# function that will return a fitted K neighbors model based on which number of neighbors was the most accurate will also return ideal num of neighbors and the accuracy of the model
def best_model(data, labels, test_percent=0.2, max_neighbors=101, print_graph=True):
        
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_percent, random_state=1)
        
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_data)
    scaled_test = scaler.transform(test_data)

    best_classifier = None
    best_accuracy = 0
    best_n_num = 0 

    if print_graph:
        x_val = range(1,max_neighbors)
        y_val = []
        
        for i in range(1, max_neighbors):

            knn_classifier = KNeighborsClassifier(n_neighbors=i)
            knn_classifier.fit(scaled_train, train_labels)
            predictions = knn_classifier.predict(scaled_test)
            accuracy = accuracy_score(predictions, test_labels) * 100
            y_val.append(accuracy)
                
            if accuracy > best_accuracy: 
                best_accuracy = accuracy
                best_n_num = i
                best_classifier = knn_classifier

        # making the graph
        plt.plot(x_val, y_val)
        plt.xlabel('Num of Neighbors')
        plt.ylabel('Accuracy Percentage')
        plt.title('Num of Neighbors vs Accuracy Percentage')
        plt.show()

    else:
        for i in range(1, max_neighbors):
            knn_classifier = KNeighborsClassifier(n_neighbors=i)
            knn_classifier.fit(scaled_train, train_labels)
            predictions = knn_classifier.predict(scaled_test)
            accuracy = accuracy_score(predictions, test_labels) * 100
                
            if accuracy > best_accuracy: 
                best_accuracy = accuracy
                best_n_num = i
                best_classifier = knn_classifier
            
    return best_classifier, best_accuracy, best_n_num


def main():
    profile_data = pd.read_csv('profiles.csv')

    column_array = ['diet', 'drinks', 'drugs', 'sex', 'smokes', 'income', 'body_type']

    replacement_value_arr = [
        # diet
        {'mostly anything': 1, 'anything': 2, 'strictly anything': 3, 'mostly vegetarian': 4, 'mostly other': 5, 'strictly vegetarian': 6, 'vegetarian': 7, 'strictly other': 8, 'mostly vegan': 9, 'other': 10, 'strictly vegan': 11, 'vegan': 12, 'mostly kosher': 13, 'mostly halal': 14, 'strictly halal': 15, 'strictly kosher': 16, 'halal': 17, 'kosher': 18},
        # drinks
        {'not at all': 1, 'rarely': 2, 'socially': 3, 'often': 4, 'very often': 5, 'desperately': 6},
        # drugs
        {'never': 1, 'sometimes': 2, 'often': 3},
        # sex
        {'m': 1, 'f': 0},
        # smokes
        {'no': 1, 'sometimes': 2, 'when drinking': 3, 'yes': 4, 'trying to quit': 5},
        # income
        {-1: 0},
        # body type
        {'average': 1, 'fit': 2, 'athletic': 3, 'thin': 4, 'curvy': 5, 'a little extra': 6, 'skinny': 7, 'full figured': 8, 'overweight': 9, 'jacked': 10, 'used up': 11, 'rather not say': 12}
        ]

    
    data = format_data(profile_data)
    
    # removing the rows where the 'body_type' value is nan
    data.remove_rows('body_type')
    
    # removing all nan values
    formatted_data = data.remove_nan_replace(column_array, replacement_value_arr)

    knn_data = formatted_data[['diet', 'drinks', 'drugs', 'sex', 'smokes', 'income']]
    knn_labels = formatted_data['body_type']

    best_classifier, accuracy, best_n = best_model(knn_data, knn_labels, print_graph=True)
    print(best_classifier)
    print('Best Number of Neighbors: {best_n} neighbors'.format(best_n=best_n))
    print('Best Accuracy: {accuracy} percent'.format(accuracy=accuracy))


main()