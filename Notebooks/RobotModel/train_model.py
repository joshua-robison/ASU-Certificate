# -*- coding: utf-8 -*-
"""
==============================================================================
Neural Network for Collision Prediction
    
    Functions:
        
        loss     -> cost_function
        visuals  -> plot_loss, plot_acc
        training -> train
        
        wrapper  -> process: train
        
This file contains functions for training and saving our model.

"""
#-----------------------------------------------------------------------------
# Imports
import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from network import NeuralNetwork
from make_datasets import generate_datasets

#-----------------------------------------------------------------------------
# Settings
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

#-----------------------------------------------------------------------------
# Loss Definition
def cost_function(prediction, target):
    loss = F.cross_entropy(prediction, target)
    
    return loss

#-----------------------------------------------------------------------------
# Training Process
def train(epoch, model, train_loader, optimizer):
    
    # activate training mode
    model.train()
    torch.set_grad_enabled(True)
    
    total_loss = 0
    correct = 0
    
    # iterate over mini-batches
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # reinitialize gradients to zero
        optimizer.zero_grad()
        
        # forward propagation
        prediction = model(data)
        
        # compute loss w.r.t targets
        loss = cost_function(prediction, target)
        
        # execute backpropagation
        loss.backward()
        
        # execute optimization step
        optimizer.step()
        
        # accumulate loss
        total_loss += loss.item() * len(data)
        
        # compute number of correct predictions
        _, preds = torch.max(prediction, dim=1)
        correct += preds.eq(target.view_as(preds)).sum().item()
        
    # compute average cost per epoch
    mean_loss = total_loss / len(train_loader.dataset)
    
    # compute accuracy
    acc = correct / len(train_loader.dataset)
    
    # display
    print('Train Epoch: {}   Avg_Loss: {:.5f}   Acc: {}/{} ({:.3f}%)'.format(
          epoch, mean_loss, correct, len(train_loader.dataset), 100. * acc))
    
    # return the average loss and the accuracy
    return mean_loss, acc
    
#-----------------------------------------------------------------------------
# Repetitive Training and Validation
def process(numEpochs, batch_size, model, optimizer, train_loader, test_loader):
    
    # cost accumulators
    train_losses = []
    eval_losses = []
    
    # performance accumulators
    train_accuracies = []
    eval_accuracies = []
    
    # learning loop
    for epoch in range(1, numEpochs + 1):
        
        # train model
        train_loss, train_acc = train(epoch, model, train_loader, optimizer)
        
        # evaluate model
        eval_loss, eval_acc = model.evaluate(model, test_loader, cost_function)
        
        # save costs
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        
        # save performances
        train_accuracies.append(train_acc)
        eval_accuracies.append(eval_acc)
        
    print('\n\n\nOptimization Ended.\n')
    
    # plot performances
    plot_loss(numEpochs, train_losses, eval_losses)
    plot_acc(numEpochs, train_accuracies, eval_accuracies)
    
    # save model
    torch.save(model.state_dict(), 'saved/saved_model.pkl', _use_new_zipfile_serialization=False)
    
#-----------------------------------------------------------------------------
# Plotting Tools
def plot_loss(epochs, training_loss, eval_loss):
    
    plt.subplot(121)
    
    x = list(range(1, epochs + 1))
    plt.plot(x, training_loss, 'b-', label='Training')
    plt.plot(x, eval_loss, 'r-', label='Validation')
    
    plt.title('Loss Evaluation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.tight_layout()

def plot_acc(epochs, training_accuracies, eval_accuracies):
    
    plt.subplot(122)
    
    x = list(range(1, epochs + 1))
    plt.plot(x, training_accuracies, 'b-', label='Training')
    plt.plot(x, eval_accuracies, 'r-', label='Validation')
    
    plt.title('Accuracy Evaluation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.tight_layout()

#-----------------------------------------------------------------------------
# Main Method
def main():
    # parameters
    batch_size = 16
    epochs = 100
    
    # initialize model
    model = NeuralNetwork()
    
    # view model
    print(model)
    
    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # generate data
    train_loader, test_loader = generate_datasets(batch_size)
    
    # run procedure
    process(epochs, batch_size, model, optimizer, train_loader, test_loader)

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    main()

