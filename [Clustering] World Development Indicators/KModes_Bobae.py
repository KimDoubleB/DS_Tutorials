# k - mode
import pandas as pd
import numpy as np

class KModes:    
    # constructor
    def __init__(self, n_clusters=8, max_iter=300, random_state=0):
        if(n_clusters < 1 or max_iter < 1):
            raise ValueError("Value of parameter is wrong.")
            
        # set parameters
        self.n_clusters = n_clusters   
        self.max_iter = max_iter
        self.random_state = random_state
                
    def HammingDistance(self, mode, data):
        # prevent error
        if(len(mode.index)!= len(data.index)):
            raise ValueError("lengths are different.\n", mode, data)
        if((mode.index != data.index).all()):
            raise ValueError("Indexes are different.\n", mode, data)
        
        result = 0
        # get distance in each columns
        for index in mode.index:
            if mode[index] != data[index]:
                result += 1
        return result
    
    # fit function
    def fit(self, fit_data):
        # prevent error
        if type(fit_data) != type(pd.DataFrame()):
            fit_data = pd.DataFrame(fit_data)

        self.fit_data = fit_data
    
        # initial modes
        self.cluster_centers_ = fit_data.sample(n=self.n_clusters,random_state=self.random_state).reset_index(drop=True)
        self.labels_ = []
        
        # ~ max_iter
        for i in range(self.max_iter):          
            new_clusters = [] # list of new clusters
            
            # cluster calculation by max_iter value
            for index, row in self.fit_data.iterrows():
                dists = []
                for index_mode, row_mode in self.cluster_centers_.iterrows():
                    # calculate distnace using hamming distance function
                    dists.append([self.HammingDistance(row_mode, row), index_mode])
                # Assign the cluster in mode with a minimum distance from the current record.
                new_clusters.append(min(dists)[1])

            if len(self.labels_) != 0 and self.labels_ == new_clusters:
                # If there is no change in the cluster, end the function.
                return self
            else:
                # If there is change in the cluster, Get new modes.
                self.labels_ = new_clusters
                self.fit_data['cluster__'] = pd.Series(self.labels_, index=self.fit_data.index)
                new_modes = pd.DataFrame(columns=fit_data.columns)
                for j in range(self.n_clusters):
                    # mode value of cluster data --> new mode
                    new_modes = new_modes.append(self.fit_data[self.fit_data.loc[:, 'cluster__'] == j].mode().iloc[0], ignore_index=True)
                
                self.cluster_centers_ = new_modes.drop('cluster__', axis=1)
                self.fit_data.drop('cluster__', axis=1, inplace=True)
            
        return self
    
    # predict
    def predict(self, pre_data):
        # prevent error
        if type(pre_data) != type(pd.DataFrame()):
            pre_data = pd.DataFrame(pre_data, columns=self.cluster_centers_.columns)

        results = []
        # cluster calculation by max_iter value
        for index, row in pre_data.iterrows():
            dists = []
            for index_mode, row_mode in self.cluster_centers_.iterrows():
                # calculate distnace using hamming distance function
                dists.append([self.HammingDistance(row_mode, row), index_mode])
            # Assign the cluster in mode with a minimum distance from the current record.
            results.append(min(dists)[1])
        self.labels_ = results
        return results

    def fit_predict(self, fit_pre_data):
        return self.fit(fit_pre_data).predict(fit_pre_data)

    def purity(self, class_data):
        cluster_data = self.labels_
        # data [class, cluster]
        df_p = pd.DataFrame({'class':class_data.tolist(), 'cluster':cluster_data})
        equal_max_class = 0
        
        # get purity value
        for i in pd.unique(df_p.cluster):
            temp = df_p[df_p['cluster'] == i]
            class_of_cluster = temp.mode()['class'][0]
            print('cluster {} - class {}' .format(i, class_of_cluster))
            equal_max_class += len(temp[temp['class']==class_of_cluster])
        purity = equal_max_class / len(self.labels_)
        return purity
