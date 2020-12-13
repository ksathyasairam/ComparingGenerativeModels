import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from PIL import Image
from IPython.display import display
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from PIL import Image
from IPython.display import display
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GeneratorModel(nn.Module):
    def __init__(self, gen_inp_dim, gen_out_dim):
        super(GeneratorModel, self).__init__()
        self.l1 = nn.Linear(gen_inp_dim, 256)
        self.l2 = nn.Linear(self.l1.out_features, self.l1.out_features*2)
        self.l3 = nn.Linear(self.l2.out_features, self.l2.out_features*2)
        self.l4 = nn.Linear(self.l3.out_features, gen_out_dim)

    def forward(self, x):
        x = F.leaky_relu(self.l1(x), 0.2)
        x = F.leaky_relu(self.l2(x), 0.2)
        x = F.leaky_relu(self.l3(x), 0.2)
        return torch.tanh(self.l4(x))
class DiscriminatorModel(nn.Module):
    def __init__(self, dis_inp_dim):
        super(DiscriminatorModel, self).__init__()
        self.l1 = nn.Linear(dis_inp_dim, 1024)
        self.l2 = nn.Linear(self.l1.out_features, self.l1.out_features//2)
        self.l3 = nn.Linear(self.l2.out_features, self.l2.out_features//2)
        self.l4 = nn.Linear(self.l3.out_features, 1)

    def forward(self, x):
        x = F.leaky_relu(self.l1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.l2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.l3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.l4(x))
latent_dim = 100
mnist_dim = train_data.train_data.size(1) * train_data.train_data.size(2)
generator = GeneratorModel(gen_inp_dim = latent_dim, gen_out_dim = mnist_dim).to(device)
discriminator = DiscriminatorModel(mnist_dim).to(device)
loss_function = nn.BCELoss()
lr = 0.0002
generator_optimiser = optim.Adam(generator.parameters(), lr = lr)
discriminator_optimiser = optim.Adam(discriminator.parameters(), lr = lr)

def train_discriminator(v):
    discriminator.zero_grad()
    r_x, r_y = v.view(-1, mnist_dim), torch.ones(batch_size, 1)
    r_x, r_y = Variable(r_x.to(device)), Variable(r_y.to(device))
    output = discriminator(r_x)
    real_loss = loss_function(output, r_y)
    real_score = output
    z = Variable(torch.randn(batch_size, latent_dim).to(device))
    f_x, f_y = generator(z), Variable(torch.zeros(batch_size, 1).to(device))
    output = discriminator(f_x)
    fake_loss = loss_function(output, f_y)
    fake_score = output
    loss = fake_loss + real_loss
    loss.backward()
    discriminator_optimiser.step()
    return  loss.data.item()

def train_generator(v):
    generator.zero_grad()
    z = Variable(torch.randn(batch_size, latent_dim).to(device))
    y = Variable(torch.ones(batch_size, 1).to(device))
    output = generator(z)
    discriminator_output = discriminator(output)
    loss = loss_function(discriminator_output, y)
    loss.backward()
    generator_optimiser.step()
    return loss.data.item()

def save_and_print_image(id):
    with torch.no_grad():
        test_z = Variable(torch.randn(batch_size, latent_dim).to(device))
        generated = generator(test_z)
        filename = 'epoch_'+str(id)+ '.jpg'
        save_image(generated.view(generated.size(0), 1, 28, 28), filename)
        pil_im = Image.open(filename)
        display(pil_im)

current_epoch = 0
no_of_epoch = 5
saving_freq = 1

for epoch in range(1, no_of_epoch+1):
    discriminator_loss_arr, generator_loss_arr = [], []
    for batch_idx, (x, _) in enumerate(train_loader):
        discriminator_loss_arr.append(train_discriminator(x))
        generator_loss_arr.append(train_generator(x))
    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % ((epoch), no_of_epoch, torch.mean(torch.FloatTensor(discriminator_loss_arr)), torch.mean(torch.FloatTensor(generator_loss_arr))))
    current_epoch+=1
    if current_epoch%saving_freq==0:
        save_and_print_image(current_epoch)
