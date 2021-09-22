import torch.nn as nn
import torch.nn.functional as F
import torch 
from scipy.sparse import diags
from sklearn.metrics import confusion_matrix

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()
            
            
def spectrum_score(confm_sum):
        array_1 = [0]
        array_2 = [0]
        for i in range(confm_sum.shape[0]):
            array_1.append(i+1)
            array_1.insert(0,i+1)
            array_2.append(i+1)
            array_2.insert(0,0-(i+1))
        diagonal = diags(array_1, array_2, shape=(confm_sum.shape[0],confm_sum.shape[0])).toarray()
        total = sum(sum(confm_sum))
        norm_confm_sum = confm_sum/total
        
        spectrum_matrix = norm_confm_sum * diagonal
        spectrum_score = float(np.sum(spectrum_matrix))
        # spectrum_score_norm = spectrum_score / i 
        # print('spectrum score: {:0.4f}'.format(spectrum_score_norm))
        return spectrum_score
        
class spectrum_loss(nn.Module):
    def __init__(self):
        super(spectrum_loss, self).__init__()

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets)
        confm = confusion_matrix(inputs, targets)
        ss_loss = spectrum_score(confm)
        spectrum_loss = 0.5*ce_loss+0.5*ss_loss
        
        return spectrum_loss
        
def get_scectrum_score(inputs, targets):
    confm = confusion_matrix(inputs, targets)
    return spectrum_score(confm_sum)
