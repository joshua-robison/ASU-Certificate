# -*- coding: utf-8 -*-
"""
==============================================================================
Neural Network for Collision Prediction
    
    Classes:
        
        NeuralNetwork

This file defines our neural network model.

"""
#-----------------------------------------------------------------------------
# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

#-----------------------------------------------------------------------------
# Neural Network Architecture
class NeuralNetwork(nn.Module):
    
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.fc1 = nn.Linear(6, 20)
        self.fc2 = nn.Linear(20, 40)
        self.fc3 = nn.Linear(40, 20)
        self.fc4 = nn.Linear(20, 2)
        
        self.dropout = nn.Dropout(p=0.50)
    
    def forward(self, x):
        
        # feed-forward
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        output = F.softmax(self.fc4(x), dim=1)
        
        return output
    
    def evaluate(self, model, eval_loader, cost_function):
        
        # activate evaluation mode
        model.eval()
        
        # initialize trackers
        total_loss = 0
        correct = 0
        
        # ignore gradients
        torch.set_grad_enabled(False)
        
        # iterate over batches
        for batch_idx, (data, target) in enumerate(eval_loader):
            
            # forward propagate
            prediction = model(data)
            
            # compute loss w.r.t the targets
            loss = cost_function(prediction, target)
            
            # accumulate loss
            total_loss += loss.item() * len(data)
            
            # compute number of correct predictions
            _, preds = torch.max(prediction, dim=1)
            correct += preds.eq(target.view_as(preds)).sum().item()
            
        # compute average cost per epoch
        mean_loss = total_loss / len(eval_loader.dataset)
        
        # compute the accuracy
        acc = correct / len(eval_loader.dataset)
    
        print('Eval:  Avg_Loss: {:.5f}   Acc: {}/{} ({:.3f}%)'.format(
            mean_loss, correct, len(eval_loader.dataset), 100. * acc))
    
        # return the average loss and the accuracy
        return mean_loss, acc

#-----------------------------------------------------------------------------
# Main Method
def main():
    # initialize model
    model = NeuralNetwork()
    
    # view model
    print(model)

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    main()

