import os
import numpy as np
from utils.utils import Logger, AverageMeter, time_to_str
from torchvision import transforms
from dataset import dataset_processing
from transforms.affine_transforms import RandomRotate
from torch.utils.data import DataLoader
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from timeit import default_timer as timer
import torchvision.models as models
from utils.report import report_precision_se_sp_yi, report_mae_mse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
INDEX = ["0", "1", "2", "3", "4"]      # ["0", "1", "2", "3", "4"]  cross_validation_index

DATASET = "ACNE04"
NET = "vgg16"
# Hyper Parameters
BATCH_SIZE = 16
BATCH_SIZE_TEST = 16
LR = 0.001              # learning rate
NUM_WORKERS = 12
NUM_CLASSES = 4
lr_steps = [30, 60, 90, 120]

SCALE_SIZE = (256, 256)
INPUT_SIZE = (224, 224)
N_SEVERITY = 4

MODEL_STATE_PATH = "./model_state/" + DATASET + "/vgg+skin_attention+augmentation+skin_color/"
LOG_FILE_NAME = './logs/' + DATASET + "/vgg+skin_attention+augmentation+skin_color/"

DATA_PATH = '../data/ACNE04/image_attention/'

np.random.seed(42)

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.feature = vgg16.features
        self.pool = vgg16.avgpool
        self.linear = vgg16.classifier
        self.output = nn.Sequential(nn.ReLU(), nn.Linear(1003, N_SEVERITY))


    def forward(self, x, color):
        x = self.feature(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = torch.cat((color, x), dim=1)
        x = self.output(x)
        return x


def train_test(cross_val_index):

    log_file = LOG_FILE_NAME + "log_%s_index%s.log" % (NET, cross_val_index)
    model_state_path = MODEL_STATE_PATH + "state_dict_index%s.pkl" % ("_".join(cross_val_index))
    log = Logger()
    log.open(log_file, mode="a")

    TRAIN_FILE = '../data/ACNE04/NNEW_trainval_' + cross_val_index + '.txt'
    TEST_FILE = '../data/ACNE04/NNEW_trainval_' + cross_val_index + '.txt'

    normalize = transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                                     std=[0.2814769, 0.226306, 0.20132513])

    dset_train = dataset_processing.DatasetProcessing(
        DATA_PATH, TRAIN_FILE, transform=transforms.Compose([
                transforms.Scale(SCALE_SIZE),
                transforms.RandomCrop(INPUT_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                RandomRotate(rotation_range=20),
                normalize,
            ]))

    dset_test = dataset_processing.DatasetProcessing(
        DATA_PATH, TEST_FILE, transform=transforms.Compose([
                transforms.Scale(INPUT_SIZE),
                transforms.ToTensor(),
                normalize,
            ]))

    train_loader = DataLoader(dset_train,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=True)

    test_loader = DataLoader(dset_test,
                             batch_size=BATCH_SIZE_TEST,
                             shuffle=False,
                             num_workers=NUM_WORKERS,
                             pin_memory=True)

    cnn = Cnn()
    cnn = cnn.cuda()
    cudnn.benchmark = True

    optimizer = torch.optim.SGD(params=cnn.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)

    loss_func = nn.CrossEntropyLoss().cuda()

    def adjust_learning_rate_new(optimizer, decay=0.5):
        """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = decay * param_group['lr']

    # training and testing
    start = timer()

    max_acc = 0
    best_report = ""

    for epoch in range(lr_steps[-1]):#(EPOCH):#
        if epoch in lr_steps:
            adjust_learning_rate_new(optimizer, 0.5)
        # scheduler.step(epoch)

        losses = AverageMeter()
        # '''
        cnn.train()
        for step, (b_x, b_y, b_color) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
            b_x = b_x.cuda()
            b_color = b_color.cuda()
            b_y = b_y.cuda()
            # train
            cnn.train()
            b_pre = cnn(b_x, b_color)
            loss = loss_func(b_pre, b_y)
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            losses.update(loss.item(), b_x.size(0))
        message = '%s  | %0.3f | %0.3f | %s\n' % (
                "train", epoch,
                losses.avg,
                time_to_str((timer() - start), 'min'))

        log.write(message)
        # '''
        if True:
            with torch.no_grad():
                test_loss = 0
                test_corrects = 0
                y_true = np.array([])
                y_pred = np.array([])

                cnn.eval()
                for step, (test_x, test_y, test_color) in enumerate(test_loader):   # gives batch data, normalize x when iterate train_loader

                    test_x = test_x.cuda()
                    test_color = test_color.cuda()
                    test_y = test_y.cuda()
                    y_true = np.hstack((y_true, test_y.data.cpu().numpy()))

                    cnn.eval()

                    b_pre = cnn(test_x, test_color)

                    loss = loss_func(b_pre, test_y)
                    test_loss += loss.data

                    _, preds = torch.max(b_pre, 1)
                    y_pred = np.hstack((y_pred, preds.data.cpu().numpy()))

                    batch_corrects = torch.sum((preds == test_y)).data.cpu().numpy()
                    test_corrects += batch_corrects

                # test_loss = test_loss.float() / len(test_loader)
                test_acc = test_corrects / len(test_loader.dataset)#3292  #len(test_loader)

                _, _, pre_se_sp_yi_report = report_precision_se_sp_yi(y_pred, y_true)

                if test_acc > max_acc:
                    max_acc = test_acc
                    best_report = str(pre_se_sp_yi_report) + "\n"
                    torch.save(cnn.state_dict(), model_state_path)
                if True:
                    log.write(str(pre_se_sp_yi_report) + '\n')
                    log.write("best result until now: \n")
                    log.write(str(best_report) + '\n')
    log.write("best result: \n")
    log.write(str(best_report) + '\n')
    return max_acc



cross_val_lists = INDEX
for cross_val_index in cross_val_lists:
    max_acc = train_test(cross_val_index)

