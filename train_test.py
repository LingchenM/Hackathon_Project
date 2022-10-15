import torch
from torch.nn import Module
from dataloader import trainLoader, testLoader
import torchvision
from torch.utils.data import DataLoader
from network import NN_testMod
from torch.utils.tensorboard.writer import SummaryWriter



#Configure hardware
device = torch.device('cuda')

#Prepare dataset
train_data = trainLoader(transform=torchvision.transforms.ToTensor())
test_data = testLoader(transform=torchvision.transforms.ToTensor())

#Load dataset
train_dataloader = DataLoader(train_data, batch_size=1)
test_dataloader = DataLoader(test_data, batch_size=1)

#Create neural network
test_mod = NN_testMod()
test_mod = test_mod.to(device)

#Create loss function
loss_func = torch.nn.CrossEntropyLoss()
loss_func = loss_func.to(device)

#Configure optimizer
learning_rate = 0.0001
optim = torch.optim.SGD(test_mod.parameters(), lr=learning_rate)

#Configure SummaryWriter
writer = SummaryWriter("NN_test_logs")


# Configure network parameters
total_train_step = 0
total_test_step = 0
epoch = 10

#Training
for i in range(epoch):
    print(f'----------Training round no. {i} begins----------')

    #Start training
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = test_mod(imgs)
        loss = loss_func(output, targets)

        #Optimization
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step += 1
        if total_train_step % 1000 == 0:
            print(f"Iterations：{total_train_step}, loss: {loss}")
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    #Start testing
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = test_mod(imgs)
            loss = loss_func(output, targets)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print(f'Test loss: {total_test_loss}')
    print(f'Test accuracy: {total_accuracy / len(test_data)}')
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy / len(test_data), total_test_step)
    total_test_step += 1

    #Save checkpoint
    torch.save(test_mod, f'D:\Desktop\Hack Project\hackathon\checkpoints\\result{i}.pth')

writer.close()



