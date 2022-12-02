import numpy as np 
import torch
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from kernels import *
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle


class EarlyStoppingModule(object):
  """
    Module to keep track of validation score across epochs
    Stop training if score not imroving exceeds patience
  """  
  def __init__(self, args):
    """
      input : args 
      patience: number of epochs to wait for improvement in validation score
      delta: minimum difference between two validation scores that can be considered as an improvement 
      best_score: keeps track of best validation score observed till now (while training)
      num_bad_epochs: keeps track of number of training epochs in which no improvement has been observed
      should_stop_now: boolean flag deciding whether training should early stop at this epoch
    """
    self.args = args
    self.patience = args.patience 
    self.delta = args.delta
    self.best_score = None
    self.num_bad_epochs = 0 
    self.should_stop_now = False

  def save_best_model(self, model, epoch): 
    fname =f"./{self.args.model_type}_bestValModel.pkl"
    pickle.dump(model, open(fname,"wb"))
    print(f"INFO:: Saving best validation model at epoch {epoch}")
    

  def load_best_model(self):
    fname =f"./{self.args.model_type}_bestValModel.pkl"
    try: 
      model = pickle.load(open(fname,"rb"))
      print(f"INFO:: Loading best validation model from {fname}")
    except Exception as e:
      print(f"INFO:: Cannot load best model due to exception: {e}") 

    return model

  def check(self, curr_score, model, epoch) :
    """Checks whether the current model has the best validation accuracy and decides to stop or proceed.
    If the current score on validation dataset is the best so far, it saves the model weights and bias.

    Args:
        curr_score (_type_): Score of the current model
        model (_type_): Trained Logistic/Linear model
        epoch (_type_): current epoch

    Returns:
        self.stop_now: Whether or not to stop

    Task1: Check for stoppage as per the early stopping criteria 
    Task2: Save best model as required
    """    

    ## TODO
    if self.best_score == None :
        self.save_best_model(model,epoch)
        self.best_score = curr_score
        self.num_bad_epochs = 0
    elif curr_score > self.best_score + self.delta :
        self.save_best_model(model,epoch)
        self.best_score = curr_score
        self.num_bad_epochs = 0
    else :
        self.num_bad_epochs += 1
        if self.num_bad_epochs > self.patience :
            self.should_stop_now = True
    ## END TODO
    return self.should_stop_now 

def train(args, Xtrain, Ytrain, Xval, Yval, model ):
    """
      tr_dataset : Num training samples * feature_dimension
      Trains for fixed number of epochs
      Keeps track of training loss and validation accuracy
    """
    es = EarlyStoppingModule(args)
    tr_dataset = data_utils.TensorDataset(Xtrain, Ytrain)
    loader = data_utils.DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataset = data_utils.TensorDataset(Xval, Yval)
    eval_loader = data_utils.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    # build model
    opt = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()),\
                           lr=args.lr, weight_decay=args.weight_decay)
    losses = []
    val_accs = []
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch[0])
            label = batch[1].float()
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        losses.append(total_loss)
        val_acc = evaluate(eval_loader, model)
        val_accs.append(val_acc)
        print("Epoch ", epoch, "Loss: ", total_loss, "Val Acc.: ", val_acc)
        if es.check(val_acc,model,epoch) :
            break

    # plt.plot(range(len(val_accs)),val_accs)
    # plt.show()
    # plt.plot(range(len(losses)),losses)
    # plt.show()
    return val_accs, losses


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


def set_seed(x=4):
    # Set random seeds
    seed = x
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)


def plot(val_accs, losses):
    plt.figure(figsize=(14,6))

    plt.subplot(1, 2, 1)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Val Accuracy", fontsize=18)
    plt.plot(val_accs)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.subplot(1, 2, 2)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Train Loss", fontsize=18)
    plt.plot(losses)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig("Training.png")



class ModelClass(torch.nn.Module):
    def __init__(self, args,input_dim):
        super(ModelClass, self).__init__()
        self.args  = args
        self.input_dm = input_dim

        self.linear_layers_one = torch.nn.Linear(input_dim,20)
        self.linear_layers_two = torch.nn.Linear(20,10)
        self.linear_layers_three = torch.nn.Linear(10,1)
        self.Loss = torch.nn.BCELoss()
        

    def forward(self, data):

        prob = torch.nn.ReLU()(self.linear_layers_one(torch.tensor(data)))
        # prob[prob<0] = 0
        prob = torch.nn.ReLU()(self.linear_layers_two(prob))
        prob = self.linear_layers_three(prob)
        pred = torch.sigmoid(prob)

        return pred

    def loss(self, pred, label):

        pred = pred.flatten()
        # batch_size = pred.shape[0]

        # loss = (-1/batch_size)*(torch.sum(torch.log(0.0001 + pred[label==1])) + torch.sum(torch.log(1.0001 - pred[label==0])))
        loss = self.Loss(pred,label)
        return loss

def evaluate(loader, model):
    model.eval() # This enables the evaluation mode for the model

    eval_score = 0
    for data in loader:
        with torch.no_grad():
            pred = model(data[0])
            label = data[1]
            pred = pred.flatten()
            pred[pred<=0.5] = 0
            pred[pred>0.5] = 1
            diff = pred - label
            diff = diff[diff == 0]
            correct = diff.shape[0]
            total = label.shape[0]
            eval_score = correct/total
    
    return eval_score

if __name__ == '__main__':
    set_seed(4)
      
    df=pd.read_csv('../../data/VoiceGender/voice.csv') ## give path of the csv file

    data = np.array(df)
    y = np.array(data[:,-1])
    x = np.array(data[:,:-1], dtype = np.float64)
    y[y == "male"] = 0
    y[y == "female"] = 1
    y = np.array(y, dtype = np.int16)

    Xtrain, Xval, Ytrain, Yval = train_test_split(x, y, test_size=0.3, random_state=42)
    Xtrain = torch.from_numpy(Xtrain)
    Ytrain = torch.from_numpy(Ytrain)

    Xval = torch.from_numpy(Xval)
    Yval = torch.from_numpy(Yval)

    Xtrain = Xtrain.float()
    Ytrain = Ytrain.long()
    Xval = Xval.float()
    Yval = Yval.long()

    args = {'batch_size': 64,
            'epochs': 600, 
            'opt': 'adam',
            'patience' : 100,
            'weight_decay': 5e-3,
            'delta' : 1e-3,
            'lr': 0.0002,
            'model_type': 'nll'} 

    args = objectview(args)

    input_dim = Xtrain.shape[1]
    my_model = ModelClass(args, input_dim)

    val_accs, losses =  train(args, Xtrain, Ytrain, Xval, Yval, my_model)
    plot(val_accs, losses)

    SVM_classifier = SVC(kernel = laplace_kernel)
    SVM_classifier.fit(Xtrain, Ytrain)
    SVM_predictions = SVM_classifier.predict(Xval)
    SVM_accuracy = np.sum(SVM_predictions == Yval.numpy())*100/Yval.shape[0]
    print("Accuracy of SVM with Laplace Kernel : " + str(SVM_accuracy))

    DecisionTree = DecisionTreeClassifier(random_state=0)
    DecisionTree.fit(Xtrain, Ytrain)
    DecisionTree_predictions = DecisionTree.predict(Xval)
    DecisionTree_accuracy = np.sum(DecisionTree_predictions == Yval.numpy())*100/Yval.shape[0]
    print("Accuracy of Decision Tree : " + str(DecisionTree_accuracy))

    RandomForest = RandomForestClassifier(n_estimators=5, random_state=0)
    RandomForest.fit(Xtrain, Ytrain)
    RandomForest_predictions = RandomForest.predict(Xval)
    RandomForest_accuracy = np.sum(RandomForest_predictions == Yval.numpy())*100/Yval.shape[0]
    print("Accuracy of Random Forest : " + str(RandomForest_accuracy))

    




