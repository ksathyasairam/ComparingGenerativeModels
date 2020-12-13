import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from PIL import Image
from IPython.display import display

def save_and_print_image(epoch):
  with torch.no_grad():
    z = torch.randn(64, 2).cuda()
    sample = vae.decoder(z).cuda()
    filename = 'epoch_'+str(id)+'.jpg'
    save_image(sample.view(64, 1, 28, 28), filename)
    pil_im = Image.open(filename)
    display(pil_im)


batch_size=100
shuffle_train_data=True
shuffle_test_data=False

train = datasets.MNIST(root='./mnist_data/',train=True,transform=transforms.ToTensor(),download=True)
test = datasets.MNIST(root='./mnist_data/',train=False,transform=transforms.ToTensor(),download=False)

train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=shuffle_train_data)

test_loader = torch.utils.data.DataLoader(dataset=test,batch_size=batch_size,shuffle=shuffle_test_data)


class VAE(nn.Module):
    def __init__(self, inp_dim, hid_l1_dim, hid_l2_dim, latent_dim):
        super(VAE, self).__init__()

        # encoder part
        self.l1 = nn.Linear(inp_dim, hid_l1_dim)
        self.l2 = nn.Linear(hid_l1_dim, hid_l2_dim)
        self.l31 = nn.Linear(hid_l2_dim, latent_dim)
        self.l32 = nn.Linear(hid_l2_dim, latent_dim)
        # decoder part
        self.l4 = nn.Linear(latent_dim, hid_l2_dim)
        self.l5 = nn.Linear(hid_l2_dim, hid_l1_dim)
        self.l6 = nn.Linear(hid_l1_dim, inp_dim)

    def encoder(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l31(h), self.l32(h)

    def decoder(self, z):
        h = F.relu(self.l4(z))
        h = F.relu(self.l5(h))
        return F.sigmoid(self.l6(h))

    def generate_sample(self, mean, var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean) # return z sample

    def forward(self, x):
        mean, var = self.encoder(x.view(-1, 784))
        z = self.generate_sample(mean, var)
        return self.decoder(z), mean, var

vae = VAE(inp_dim=784, hid_l1_dim=512, hid_l2_dim=256, latent_dim=2)
if torch.cuda.is_available():
    vae.cuda()
optimizer = optim.Adam(vae.parameters())


def loss_function(reconstruction_v, v, mean, var):
    BCE = F.binary_cross_entropy(reconstruction_v, v.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())
    return BCE + KLD

def train_vae(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        reconstruction_batch, mean, var = vae(data)
        loss = loss_function(reconstruction_batch, data, mean, var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch+1, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch+1, train_loss / len(train_loader.dataset)))

def test_vae():
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda()
            reconstruction, mean, var = vae(data)
            test_loss += loss(reconstruction, data, mean, var).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

current_epoch = 0
no_of_epoch = 500
saving_freq = 50

for epoch in range(no_of_epoch):
    train_vae(epoch)
    test_vae()
    current_epoch+=1
    if current_epoch%saving_freq==0:
      save_and_print_image(current_epoch)
