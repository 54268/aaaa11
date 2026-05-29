import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def auto_make_Confusion_Matrix_graph(savedir):
    

    
    torch_matrix = torch.zeros(10, 10)
    torch_matrix = torch_matrix.to(torch.int32)

    
    my_preds_list = []
    my_targets_list = []
    
    with open(savedir + '/my_preds.txt', 'r') as file:
        
        for line in file:
            
            line = line.strip()
            
            numbers = line.split()
            
            for number in numbers:
                
                number = int(number)
                
                my_preds_list.append(number)
    with open(savedir + '/my_targets.txt', 'r') as file:
        
        for line in file:
            
            line = line.strip()
            
            numbers = line.split()
            
            for number in numbers:
                
                number = int(number)
                
                my_targets_list.append(number)

    
    for i in range(len(my_preds_list)):
        p=my_preds_list[i]
        t=my_targets_list[i]
        torch_matrix[t,p]=torch_matrix[t,p]+1

    
    


    
    cost_matrix = torch_matrix.shape[0] * torch_matrix.shape[1] - torch_matrix

    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    
    
    
    
    

    
    conf_matrix =torch_matrix[row_ind, :][:, col_ind]

    
    row_sums = conf_matrix.sum(axis=1, keepdims=True)

    
    conf_matrix = conf_matrix / row_sums

    
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    
    accuracies = conf_matrix.diag() / conf_matrix.sum(1)

    
    plt.figure(1)  
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    plt.yticks(range(len(class_names)), class_names)
    plt.xticks(range(len(class_names)), class_names)
    plt.ylabel('True Label')  
    plt.xlabel('Predicted Label')  
    
    plt.tight_layout()  

    
    thresh = conf_matrix.max() / 2
    for x in range(len(class_names)):
        for y in range(len(class_names)):
            
            info = round(conf_matrix[y, x].item(), 2)
            
            if info != 0:
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")


    
    file_name_1 = '/Confusion_Matrix.png'
    plt.savefig(savedir + file_name_1)  
    plt.show()
    plt.close(1)  

    
    plt.figure(2)  
    plt.bar(class_names, accuracies.numpy() * 100, color='grey')

    
    for i in range(len(accuracies)):
        plt.text(i, accuracies[i] * 100 + 2, f'{accuracies[i] * 100:.2f}%', ha='center', va='bottom')

    plt.ylim(0, 110)  
    plt.ylabel('Accuracy')
    plt.xlabel('ZigBee')
    plt.tight_layout()  


    
    file_name_2 = '/Probability_distribution.png'
    plt.savefig(savedir + file_name_2)  
    plt.show()
    plt.close(2)  