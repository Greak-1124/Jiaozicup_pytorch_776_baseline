import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import  pandas as pd
import  os
from tqdm import tqdm
from collections import Iterable
import numpy as np
class Agent(object):
    def __init__(self,model,device_info,save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir=save_dir
        self.device =device_info["device"] 
        # self.saver=None
        self.model=model
        self.ParallelModel = torch.nn.DataParallel(model, device_ids= device_info["device_ids"])
        self.ParallelModel.to(self.device)
    def summary(self):
        print(self.model)

    def compile(self,loss_dict,optimizer,metrics):
        self.loss_dict=loss_dict
        self.optimizer=optimizer
        self.metrics=metrics


    def fit_generator(self,dataloader,epochs, validation_data,reduceLR=None,earlyStopping=None,**kwargs):
        metric=self.metrics
        loss_dict= self.loss_dict
        valid_acc=[]
        for epoch in range(epochs):
            print("epoch:{}-lr:{:.8f}".format(epoch,self.optimizer.state_dict()['param_groups'][0]['lr'])+"-"*5)
            #train
            phase="train"
            self.model.train()
            metric.reset()
            result_epoch=self.iter_on_a_epoch(phase,dataloader,loss_dict,metric)

            # log
            s = 'phase:{}-'.format(phase)
            for key, val in result_epoch.items():
                if not isinstance(val, Iterable):
                    s += ",{}:{:.4f}".format(key, val)
            print(s)

            #valid
            phase = "valid"
            metric.reset()
            self.model.eval()
            valid_dataloader=DataLoader(validation_data,batch_size=1024,drop_last=False)
            result_epoch = self.iter_on_a_epoch( phase, valid_dataloader, loss_dict, metric)
            valid_acc.append(result_epoch["acc_metrics"])

            # log
            s = 'phase:{}---'.format( phase)
            for key, val in result_epoch.items():
                if not isinstance(val, Iterable):
                    s += ",{}:{:.4f}".format(key, val)
            print(s)


            #保存模型
            # 保存验证集准确率>0.7的当前最高准确率权重
            if (valid_acc[-1] > 0.7 and valid_acc[-1] == max(valid_acc)) or (epoch==epochs-1):
                save_name="epo_{}-score_{:.5f}.pth".format(epoch, valid_acc[-1])
                self.save_model(save_name)
            # recude lr
            if reduceLR is not  None:
                epoch_loss = sum([val for key, val in result_epoch.items() if "loss" in key])
                reduceLR.step(valid_acc[-1], epoch)

            # earlyStopping
            if earlyStopping is not None:
                earlyStopping.step()



    def iter_on_a_epoch(self, phase, dataloader,loss_dict, metric, **kwargs):
        assert  phase in ["train","valid","test"]
        result_epoch = {"count": 0,}
        metric.reset()
        # for cnt_batch, batch in zip(tqdm(range(1, len(dataloader) + 1)), dataloader):
        for cnt_batch, batch in zip(range(1, len(dataloader) + 1), dataloader):
            result_batch = self.iter_on_a_batch(batch, loss_dict=loss_dict, phase=phase)
            #返回结果
            score_batch,label_batch,img_batch=result_batch["score_batch"],result_batch["label_batch"],result_batch["img_batch"]

            metric.add_batch(label_batch.astype(np.float),score_batch.astype(np.float))
            # print(np.array(metric.labels).shape)
            # 返回损失
            result_epoch["count"] += label_batch.shape[0]
            for key, val in result_batch["loss"].items():
                key = key + "_loss"
                if key not in result_epoch.keys(): result_epoch[key] = []
                result_epoch[key].append(val)
            # ###### 打印loss
            # if phase == "train":
            #     cul_lr = self.optimizer_ft.state_dict()['param_groups'][0]['lr']
            #     s = "epoch:{},batch:{},lr:{:.5f}".format(epoch, cnt_batch, float(cul_lr))
            #     for key, loss in result["loss"].items():
            #         s += ",{}:{:.4f}".format(key, float(loss))
            #     # self.logger.info(s)
            #     print(s)

        # 将所有loss平均
        for key, val in result_epoch.items():
            if "loss" in key:
                result_epoch[key] = np.array(val).sum() / len(val)

        metric_dict=metric.apply()
        for key,val in metric_dict.items():
            key=key+"_metrics"
            result_epoch[key]=val
        return result_epoch

    def iter_on_a_batch(self, batch,  phase,loss_dict):
        assert phase in ["train", "valid", "test",],print(phase)
        # self.model.setMode("segment")
        img_tensor, label_tensor = batch
        model=self.ParallelModel
        optimizer=self.optimizer
        device=self.device
        # forward
        img_rensor = self.type_tran(img_tensor)

        label_tensor =self.type_tran(label_tensor)
        score_tensor = model(img_rensor)
        # update_mask_batch=mask_tensor.detach().cpu().numpy()
        ###### cul loss
        losses = dict()
        if phase in ["train", "valid", "test"]:
            for name,loss in loss_dict.items():
                loss_val = loss(score_tensor, label_tensor)

                losses[name] = loss_val
        ##### backward
        if phase in ["train"]:
            assert isinstance(losses, dict)
            model.zero_grad()
            loss_sum = sum(list(losses.values()))
            loss_sum.backward()
            optimizer.step()
        #### return

        score_tensor=score_tensor.softmax(dim=-1)
        img_batch = img_rensor.detach().cpu().numpy()
        label_batch = label_tensor.detach().cpu().numpy()
        score_batch = score_tensor.detach().cpu().numpy()
        result = {"img_batch": img_batch,"label_batch": label_batch, "score_batch": score_batch}
        if phase in ["train", "valid", "test"]:
            sum_loss = 0
            for key, loss in losses.items():
                losses[key] = float(loss)
                sum_loss += float(loss)
            # losses["sum"] = sum_loss
        result["loss"] = losses
        return result

    def load_weights(self,load_name):
        save_dir = self.save_dir + "/model/"
        load_path=os.path.join(save_dir,load_name)
        if os.path.exists(load_path):
            pthfile = torch.load(load_path)
            # print(pthfile.keys())
            self.model.load_state_dict(pthfile, strict=True)
            print("load weights from {}".format(load_path))
        else:
            raise  Exception("Load model falied, {} is not existing!!!".format(load_path))

    def save_model(self,save_name):
        save_dir=self.save_dir+"/model/"
        if not  os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path=os.path.join(save_dir,save_name)
        print("save weights to {}".format(save_path))
        torch.save(self.model.state_dict(),save_path)


    def load_best_model(self):

        load_names=[  name for name in os.listdir(self.save_dir+"/model/") if name.endswith(".pth")]
        load_name = sorted(load_names, key=lambda x: float(x.split(".")[-2]),
                           reverse=True)[0]
        self.load_weights(load_name)

    def predict(self,data,phase,batch_size=1024):
        # valid
        dataloader = DataLoader(data, batch_size=batch_size, drop_last=False,shuffle=False)
        score_batchs=[]
        result_epoch = {"count": 0,}
        for cnt_batch, batch in zip(tqdm(range(1, len(dataloader) + 1)), dataloader):
            result_batch = self.infer_on_a_batch(batch)
            #返回结果
            score_batch,img_batch=result_batch["score_batch"],result_batch["img_batch"]
            score_batchs.append(score_batch)
            # 返回损失
            result_epoch["count"] += score_batch.shape[0]
        dim=score_batchs[0].shape[-1]
        score_array=np.concatenate(score_batchs,axis=0)

        df = pd.DataFrame(score_array)
        df.to_csv(self.save_dir + "/{}_score.csv".format(phase))

        return score_array

    def infer_on_a_batch(self, batch):
        img_tensor = batch
        # forward
        img_rensor = self.type_tran(img_tensor)
        score_tensor = self.ParallelModel(img_rensor)
        score_tensor=score_tensor.softmax(dim=-1)
        #### return
        img_batch = img_rensor.detach().cpu().numpy()
        score_batch = score_tensor.detach().cpu().numpy()
        result = {"img_batch": img_batch, "score_batch": score_batch}
        return result


    def type_tran(self,data):

        return  data.to(torch.float32).to(self.device)

