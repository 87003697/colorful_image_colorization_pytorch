import matplotlib.pyplot as plt
import torch
class Loss_recorder():
    def __init__(self, loss_type):
        assert loss_type in ['train', 'val']
        self.type = loss_type
        self.loss_record = []
        
        if self.type == 'train':
            self.sub_loss_record = []
        
    def take(self,loss):
        if self.type == 'val':
            self.loss_record.append(loss)
        elif self.type == 'train':
            self.sub_loss_record.append(loss)
        
    def save(self,):
        
        if self.type == 'val':
            plt.plot(self.loss_record)
            plt.xlabel('epochs')
            plt.ylabel('val loss')
            plt.title('Model Convergence')
            plt.savefig('val_loss_epoches.png')
            print('val losse recorded.')
            plt.clf()

            
        else:
            self.loss_record.append(sum(self.sub_loss_record) /
                                   len(self.sub_loss_record))
            self.sub_loss_record = []
            
            plt.plot(self.loss_record)
#             plt.xlabel('per {} pochs'.format(args.log_interval))
            plt.xlabel('per {} batches'.format(200))
            plt.ylabel('train loss')
            plt.title('Model Convergence in Batches')
            plt.savefig('train_loss_batches.png')
            print('training losses recorded.')
            plt.clf()
            
    def reset(self,):
        self.loss_record = []
        
        if self.type == 'train':
            self.sub_loss_record = []