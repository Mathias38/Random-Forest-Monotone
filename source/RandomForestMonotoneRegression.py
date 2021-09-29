# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:29:11 2021

@author: mathias chastan
"""

import random
import numpy as np
import copy
from sklearn.tree import DecisionTreeRegressor

class RandomForestClassifier(object):

    """
    :param trees: decision trees
    :param nb_trees: Number of decision trees to use
    :param nb_samples: Number of samples to give to each tree
    :param max_depth: Maximum depth of the trees
    :param removed trees: count of trees removed by the monotone constraint 
    :param validation_rows: list of random rows used for the monotony test
    :param accepted_tree_percent: percentage of decision trees that passed the monotony test
    :param values: min and max of each variable
    :param dtypes: type of each variable
    """
    def __init__(self, nb_trees, max_depth=5,  max_features = 0.5):
        self.trees = []
        self.nb_trees = nb_trees
        self.max_depth = max_depth
        self.validation_rows = [[]]
        self.accepted_trees_percent = 0
        self.values = []
        self.dtypes = []
        self.max_features = max_features
    
    """
    :param data: the data set
    """          
    def fit(self, x, y):
                     
        trees = []
        for i in range(0,self.nb_trees):
            tree = DecisionTreeRegressor(max_depth = self.max_depth, max_features = self.max_features)
            trees.append(tree.fit(x, y))
            i = i + 1
        self.trees = trees    
    
    """
    :param data: the data set
    :param nb_rows: number of test rows
    :param postivies_vars: indexes of postivie monotone variables
    :param negative_vars: indexes of negative monotone variables
    :param target monotony_percent: required monotony percent required for a tree to pass the monotony test (list : one by variable)
    :param nb_tests_mono_pos: number of increment done by row for each postivie variable between min and max
    :param nb_test_mono_neg: number of increment done for each negative variable between min and max
    """
    def fit_with_monotony_drop_out(self, x, y, nb_rows, positive_vars, negative_vars, target_monotony_percent, nb_tests_mono_pos, nb_tests_mono_neg):
        
        trees =  []
        min_max_dtypes = self.get_min_max_dtypes(x)
        self.values = min_max_dtypes["min_max"]
        self.dtypes = min_max_dtypes["dtypes"]
        self.validation_rows = self.create_validation_rows(self.values, nb_rows, self.dtypes)

        c = 0
        ct = 0
        while c < self.nb_trees:
            tree = DecisionTreeRegressor(max_depth = self.max_depth, max_features = self.max_features)
            tree.fit(x, y)
            ct = ct + 1 
            if self.test_tree(self.values, tree, positive_vars, negative_vars, target_monotony_percent, nb_tests_mono_pos, nb_tests_mono_neg, self.dtypes):
                trees.append(tree)
                c = c + 1
                print(c)
        self.trees = trees    
        self.accepted_trees_percent = c / ct * 100
    
    """
    :param feature: one row of test data
    """
    def predict(self, feature):
        
        predictions = []      
        for tree in self.trees:
            predictions.append(tree.predict(np.array(feature).reshape(1,-1)))
                                                
        return np.mean(predictions)
            
    """
    :param test: test data
    """
    def pred(self, test):
        predictions = []
        for feature in test: 
            predictions.append(self.predict(feature))
            
        return predictions
    
    """
    :param values: min and max of each variable
    :param var_types: type of each variable
    :description this function creates one random row using variables types and min/max values
    """
    def create_validation_row(self, values, var_type):
        
        row = [0] * len(values)

        for i in range(0, len(row)):
            if var_type[i] == "d" :
                row[i] = random.randint(values[i][0], values[i][1])
            elif var_type[i] == "c" :
                row[i] = random.uniform(values[i][0], values[i][1])               
                
        return row
    
    """
    :param values: min and max of each variable
    :param nbrows: number of test rows
    :param var_types: type of each variable
    :description this function creates a defined number of random rows 
    """
    def create_validation_rows(self, values, nb_rows, var_types):
        
        rows = []
        for i in range(0, nb_rows):
            rows.append(self.create_validation_row(values, var_types))
        
        return rows
    
    """
    :param  x : Data
    :description this function gets min/max values and dtypes using the data set
    """
    def get_min_max_dtypes(self, x):
          
        min_max = []
        dtypes = []
        for col in x.columns:
            min_max.append([np.min(x[col]), np.max(x[col])])
            if x[col].dtype == 'int32':
                dtypes.append("d")
            else:
                dtypes.append("c")
                
        d = dict();  
        d['min_max'] = min_max
        d['dtypes'] = dtypes
        return d
    
    """
    :param sign: positive or negative monotony
    :param idx: index of tested variable
    :param var_values: min and max for the tested variable
    :param tree: tested tree
    :param nb_test_mono:  number of increment done by row for each postivie variable between min and max
    :param var_type: type of the variable variable
    :description test monotony for a given variable. The test is made by incrementing test rows using a fraction of the maximum value.
    The size of each increment depends on nb_test_mono. Then the prediction of row and incremented test_row are compared. 
    If they are monotonous, mono count is incremented. The monotony percentage is returned.
    """      
    def get_var_mono_percent(self, sign, idx, var_values, tree, nb_test_mono, var_type):
        
        count = 0
        mono_count = 0
        rows = copy.deepcopy(self.validation_rows[:])
        
        for row in rows:
            test_row = copy.deepcopy(row[:])
            
            for i in range(0, nb_test_mono):
                count = count + 1
                                
                if var_type == "d":
                    row[idx] = int(round((var_values[1] * (i / nb_test_mono)), 0))
                    test_row[idx] = int(round((var_values[1] * ((i + 1) / nb_test_mono)), 0))
                    
                elif var_type == "c":
                    row[idx] = int(round((var_values[1] * (i / nb_test_mono)), 0))
                    test_row[idx] = int(round((var_values[1] * ((i + 1) / nb_test_mono)), 0))
                    
                test_row_pred = tree.predict(np.array(test_row).reshape(1,-1))
                row_pred = tree.predict(np.array(row).reshape(1,-1))
                
                if sign == "pos" :
                    if test_row_pred >= row_pred:
                        mono_count = mono_count + 1 
                elif sign == "neg" :
                    if test_row_pred <= row_pred:
                        mono_count = mono_count + 1 
                        
        return mono_count / count * 100 
    
    """
    :param values: min and max of each variable
    :param trees: decision tress
    :param postivies_vars: indexes of postivie monotone variables
    :param negative_vars: indexes of negative monotone variables
    :param target monotony_percent: required monotony percent required for a tree to pass the monotony test (list : one by variable)
    :param nb_tests_mono_pos: number of increment done by row for each postivie variable between min and max
    :param nb_test_mono_neg: number of increment done for each negative variable between min and max
    :param var_types: type of each variable
    :description test_trees use get_var_mono_percent to test each variable of each tree. If the tree follows the monotony constraint it is accepted.
    """                   
    def test_tree(self, values, tree, positive_vars, negative_vars, target_monotony_percent, nb_tests_mono_pos, nb_tests_mono_neg, var_types):
        
        total_monotony_percent = []
        res = True
        
        for k in range(0, len(positive_vars)): 
            total_monotony_percent.append(self.get_var_mono_percent("pos", positive_vars[k], values[positive_vars[k]], tree, nb_tests_mono_pos[k], var_types[positive_vars[k]]))
        
        for h in range(0, len(negative_vars)):
            total_monotony_percent.append(self.get_var_mono_percent("neg", negative_vars[h], values[negative_vars[h]], tree, nb_tests_mono_neg[h], var_types[negative_vars[h]]))

        for i in range(0,len(total_monotony_percent)):
            if total_monotony_percent[i] < target_monotony_percent[i] :
                res = False
                
        return res
    
    """
    :param column_names: column names
    :param trees: decision tress
    :param postivies_vars: indexes of postivie monotone variables
    :param negative_vars: indexes of negative monotone variables
    :param target monotony_percent: required monotony percent required for a tree to pass the monotony test (list : one by variable)
    :param nb_tests_mono_pos: number of increment done by row for each postivie variable between min and max
    :param nb_test_mono_neg: number of increment done for each negative variable between min and max
    :param var_types: type of each variable
    :description use this function in preparation phase to get all variables monotony
    """             
    def print_all_variables_monotony(self, column_names, values, tree, positive_vars, negative_vars, target_monotony_percent, nb_tests_mono_pos, nb_tests_mono_neg, var_types) :
                
        for k in range(0, len(positive_vars)): 
            print(column_names[positive_vars[k]])
            print(self.get_var_mono_percent("pos", positive_vars[k], values[positive_vars[k]], tree, nb_tests_mono_pos[k], var_types[positive_vars[k]]))
            
        for h in range(0, len(negative_vars)):
            print(column_names[negative_vars[h]])
            print(self.get_var_mono_percent("neg", negative_vars[h], values[negative_vars[h]], tree, nb_tests_mono_neg[h], var_types[negative_vars[h]]))

              

        
            
