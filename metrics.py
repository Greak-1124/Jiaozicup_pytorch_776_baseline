import numpy as  np
from utils import  get_acc_func,get_acc_combo
acc_combo_func=get_acc_combo()
acc_func=get_acc_func()

class My_metrics(object):

    def __init__(self):
        pass

        self.labels=[]
        self.scores=[]


    def reset(self):
        self.labels=[]
        self.scores=[]

    def add_batch(self,labels,scores):
        self.labels.append(labels)
        self.scores.append(scores)

    def apply(self):

        labels=np.concatenate(self.labels,axis=0)
        scores=np.concatenate(self.scores,axis=0)
        acc_combo=acc_combo_func(labels,scores)
        acc=acc_func(labels,scores)
        return {"acc":acc,"acc_combo":acc_combo}



