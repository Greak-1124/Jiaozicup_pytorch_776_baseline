import torch
from torch import nn
import torch.nn.functional as F
class CELoss(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.nllloss=  nn.NLLLoss(reduction=reduction)
    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                             .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                             .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                             .format(x.size()))
        x = self.log_softmax(x)

        target=torch.argmax(target,dim=-1)

        loss=self.nllloss(x,target=target)

        return loss 

class Model(nn.Module):
    def __init__(self, num_classes=19):
        super(Model, self).__init__()

        # input: 1, num, features_num
     #   base_channel=64 
        self.features = nn.Sequential( 
            # 1
            nn.Conv2d(1, 32, kernel_size=(3, 3),stride=(1,1),padding=(1,1)),
        #    nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            

            nn.Conv2d(32, 64, kernel_size=(3, 3),stride=(1,1),padding=(1,1)),
        #    nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 2 此处BN收益最大
            nn.Conv2d(64, 128,kernel_size=(3, 3), stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AvgPool2d(2,2),
            # 3
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1,1),padding=(1,1)),
          #  nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          #  nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # last  
            nn.AdaptiveAvgPool2d((1,1)),
            nn.BatchNorm2d(512),
            nn.Dropout(0.1), 
        )
        self.classier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        
        x = x.view(x.shape[0], -1) 
        x = self.classier(x) 
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2d):
                # nn.init.constant_(m.weight, 0)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_normal(m.weight)


                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

