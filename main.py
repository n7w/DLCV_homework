#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from google.colab import drive
# drive.mount('/content/drive',force_remount=True)
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('cat /usr/local/cuda/version.txt')


# In[ ]:


get_ipython().system('wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb')
get_ipython().system('dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb')
get_ipython().system('apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub')
get_ipython().system('apt-get update')
get_ipython().system('apt-get install cuda=9.0.176-1')


# In[ ]:


get_ipython().system('wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb')
get_ipython().system('dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb')
get_ipython().system('apt-get update')
get_ipython().system('apt-get install cuda=8.0.61-1')


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import warnings

# path = '/content/drive/My Drive/modelnet40_normal_resampled/modelnet40_shape_names.txt'
# open(path)
print(torch.cuda.is_available())
torch.backends.cudnn.enabled = False
warnings.filterwarnings('ignore')


# In[ ]:


# data loader
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ModelNetDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                          in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        if self.uniform:
            point_set = self.farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[0:self.npoints,:]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if not self.normal_channel:
            point_set = point_set[:, 0:3]
        if len(self.cache) < self.cache_size:
            self.cache[index] = (point_set, cls)
            
        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)

    def farthest_point_sample(point, npoint):
        """
        Input:
            xyz: pointcloud data, [N, D]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [npoint, D]
        """
        N, D = point.shape
        xyz = point[:,:3]
        centroids = np.zeros((npoint,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        point = point[centroids.astype(np.int32)]
        return point

# test
# data = ModelNetDataLoader('/content/drive/My Drive/modelnet40_normal_resampled/',split='train', uniform=False, normal_channel=True,)
# DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
# for point,label in DataLoader:
#   print(point.shape)
#   print(label.shape)


# In[ ]:


# geo-cnn


# In[ ]:


# provider functions
def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
      Input:
          BxNx3 array, original batch of point clouds
      Return:
          BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc


# test function
def test(model, loader, vote_num = 0, num_class = 40):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    # points, target = points.cuda(), target.cuda()
    classifier = model.eval()
    if vote_num > 0:
        # vote_pool = torch.zeros(target.size()[0],num_class).cuda()
        vote_pool = torch.zeros(target.size()[0],num_class)
        for _ in range(vote_num):
        pred, _ = classifier(points)
        vote_pool += pred
        pred = vote_pool/vote_num
    else:
        pred, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    for cat in np.unique(target.cpu()):
        classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
        class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
        class_acc[cat,1]+=1
    correct = pred_choice.eq(target.long().data).cpu().sum()
    mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


# In[ ]:


# train
'''parameters'''
batch_size = 24
train_epoch = 200
learning_rate = 0.001
decay_rate = 1e-4
num_point = 1024
num_class = 40
data_path = '/content/drive/My Drive/modelnet40_normal_resampled/'
model_save_path = '/content/drive/My Drive/best_model.pth'
gpu = '0'

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

'''data loading'''
print('load dataset....')
train_dataset = ModelNetDataLoader(data_path,num_point,'train')
train_data_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=4,shuffle=True)
test_dataset = ModelNetDataLoader(data_path,num_point,'test')
test_data_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,num_workers=4,shuffle=False)
print('data loaded\n')

'''model loading'''
print('model loading')
classifier = Model(num_class).cuda()
criterion = Loss().cuda()

try:
    checkpoint = torch.load(model_save_path)
    start_epoch = checkpoint['epoch']
    classifier.load_state_dict(checkpoint['model_state_dict'])
    print('Use pretrain model, current epoch: %d' % (start_epoch))
except:
    print('No existing model, starting training from scratch...')
    start_epoch = 0

optimizer = torch.optim.Adam(classifier.parameters(),lr=learning_rate,betas=(0.9, 0.999),eps=1e-08,weight_decay=decay_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

global_epoch = 0
best_instance_acc = 0.0
best_class_acc = 0.0
mean_correct = []
print('model loaded\n')


# In[ ]:


'''training'''
print('start training')
for epoch in range(start_epoch, train_epoch):
    scheduler.step()
    for batch_id, data in tqdm(enumerate(train_data_loader, 0), total=len(train_data_loader), smoothing=0.9):
        points, target = data
        points = points.data.numpy()
        points = random_point_dropout(points)
        points[:,:, 0:3] = random_scale_point_cloud(points[:,:, 0:3])
        points[:,:, 0:3] = shift_point_cloud(points[:,:, 0:3])
        points = torch.Tensor(points)
        target = target[:, 0]

        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()    

        classifier = classifier.train()
        pred, trans_feat = classifier(points)
        loss = criterion(pred, target.long(), trans_feat)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
        loss.backward()
        optimizer.step()
  
    train_instance_acc = np.mean(mean_correct)
    print('Train %d Accuracy: %f' % (epoch + 1,train_instance_acc))

    with torch.no_grad():
        global_epoch += 1
        instance_acc, class_acc = test(classifier.eval(), test_data_loader)
        if instance_acc >= best_instance_acc:
        best_instance_acc = instance_acc   
        best_epoch = epoch + 1

        if class_acc >= best_class_acc:
        best_class_acc = class_acc
        if instance_acc >= best_instance_acc:
            state = {
                'epoch': best_epoch,
                'instance_acc': instance_acc,
                'class_acc': class_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
        torch.save(state, model_save_path)
        print('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

print('training fin!\n')


# In[ ]:


'''test'''
vote_num = 3
with torch.no_grad():
    instance_acc, class_acc = test(classifier.eval(), test_data_loader, vote_num = vote_num)
    print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

