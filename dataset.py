import warnings
warnings.filterwarnings("ignore")
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler

def standardization(X_df):

    X_df_cp = X_df.copy()

    not_use_cols = ['fragment_id', 'time_point', 'behavior_id', 'train_scene', 'train_action']
    use_cols = [col for col in X_df.columns if col not in not_use_cols]
    train_std = StandardScaler().fit_transform(X_df[use_cols])
    train_std = pd.DataFrame(train_std,columns=use_cols)

    for col in train_std.columns:
        X_df_cp[col] = train_std[col]

    return X_df_cp

    
    
class Make_data(object):
    def __init__(self,data_path,n_classes=19,**kwargs):       
        self.data_path = data_path
 
    def load_dataset(self):
        train = pd.read_csv(self.data_path+'sensor_train.csv')
        test = pd.read_csv(self.data_path+'sensor_test.csv')
        

        ## 获取label
        y = train.groupby('fragment_id')['behavior_id'].min()
        
        train['mod'] = (train.acc_x ** 2 + train.acc_y ** 2 + train.acc_z ** 2) ** .5
        train['modg'] = (train.acc_xg ** 2 + train.acc_yg ** 2 + train.acc_zg ** 2) ** .5

        test['mod'] = (test.acc_x ** 2 + test.acc_y ** 2 + test.acc_z ** 2) ** .5
        test['modg'] = (test.acc_xg ** 2 + test.acc_yg ** 2 + test.acc_zg ** 2) ** .5
        
        train_std = standardization(train)
        test_std = standardization(test)
      
        x = np.zeros((7292, 1, 60, 8))  
        t = np.zeros((7500, 1, 60, 8))
        
        for i in tqdm(range(7292)):
            tmp = train_std[train_std.fragment_id == i][:61] 
            x[i,0, :,:] =resample(tmp.drop(['fragment_id', 'time_point', 'behavior_id'],
                                            axis=1), 60 , np.array(tmp.time_point))[0]
        for i in tqdm(range(7500)):
            tmp = test_std[test_std.fragment_id == i][:61]
            t[i,0, :,:] = resample(tmp.drop(['fragment_id', 'time_point'],
                                            axis=1), 60,  np.array(tmp.time_point))[0]
        return x, y, t
        










