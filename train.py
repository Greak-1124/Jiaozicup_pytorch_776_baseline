from sklearn.model_selection import StratifiedKFold
from models import *
from dataset import  Make_data
import pandas as pd
from metrics import My_metrics
from torch.utils.data import DataLoader
from torch_func import  Agent
from torch.optim import SGD, lr_scheduler, Adam
from utils import *
import time
import os

DEVICE_INFO=dict(
    gpu_num=torch.cuda.device_count(),
    device_ids = range(0, torch.cuda.device_count(), 1),
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu",
    index_cuda=0, )

METRICS = My_metrics()

set_torch_seed(2020)
seeds=[2031]
proba_t = np.zeros((7500, 19))

valid = np.zeros((7292, 19))  

mk_data = Make_data(data_path='./dataset/')

x, y, t = mk_data.load_dataset()

kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state=2031) 
test_X=[t[i] for i in range(t.shape[0])]

for seed in seeds:
    for fold, (xx, yy) in enumerate(kfold.split(x, y)): 
        
        timer = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        SAVE_DIR="./save_{}/".format(timer)
        save_dir=os.path.join(SAVE_DIR,"flod_{}".format(0))
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        y_ =  one_hot(y,19)
 
        model=Model(num_classes=19) 
        agent=Agent(model=model,device_info=DEVICE_INFO, save_dir=save_dir)
    
        LOSS={ "celoss":CELoss() }

        OPTIM=Adam(model.parameters(), lr=0.001, weight_decay=0.001)

        reduceLR = lr_scheduler.ReduceLROnPlateau(OPTIM, mode="max", factor=0.5, patience=8, verbose=True)

        early_stopping = None

        agent.compile(loss_dict=LOSS, optimizer=OPTIM, metrics=METRICS)


        x_trn = x[xx]
        y_trn_c = y_[xx]
        x_vld = x[yy]
        y_vld_c = y_[yy]

        trn_data = [(x_trn[i],y_trn_c[i]) for i in range(x_trn.shape[0])]
        vld_data = [(x_vld[i],y_vld_c[i]) for i in range(x_vld.shape[0])]

        train_generator=DataLoader(trn_data, batch_size= 512, shuffle=True, num_workers=0)

        agent.fit_generator(train_generator, epochs=1, validation_data=vld_data,  reduceLR=reduceLR, earlyStopping=early_stopping)

        agent.load_best_model()
 
        scores_test= agent.predict(test_X, batch_size=1024, phase="test")
        proba_t+=scores_test/(20.*len(seeds))

        ################计算每一轮验证集的得分###############################################
        val = agent.predict(x_vld, batch_size=1024, phase="valid")
        valid[yy] += agent.predict(x_vld, batch_size=1024, phase="valid")/len(seeds)
        
        val_labels = np.argmax(val,axis=1)

        val_score = sum(My_acc_combo(y_true, y_pred, 'behavior') for y_true, y_pred in zip(y[yy].values, val_labels)) / val_labels.shape[0]
        acc_vld = round(get_acc(y[yy].values, val), 5)

        print('\n官方得分：', round(val_score, 5))
        print('准确率得分：', acc_vld)
        
## 计算总的得分
valid_labels = np.argmax(valid, axis=1)
score = sum(My_acc_combo(y_true, y_pred, 'behavior') for y_true, y_pred in zip(y, valid_labels)) / valid_labels.shape[0]
print('\n'+'比赛的评价分数：', round(score, 5))
print('\n'+'准确率：', round(get_acc(y, valid), 5))

## 保存文件成提交
sub = pd.read_csv('./dataset/result.csv')
sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv('submit.csv', index=False)

