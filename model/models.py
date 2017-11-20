import torch
import torch.nn as nn

class G(nn.Module):
    def __init__(self, nz=100, output_nc=3, ngf=64):
        super(G, self).__init__()
        '''
        self.fc1 = nn.Sequential(
            nn.Linear(nz, 8192)
        )
        self.bn = nn.BatchNorm2d(ngf*8)
        self.relu = nn.ReLU()
        '''
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, output_nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state 16*16
        )

    def forward(self, nz1, nz2):
        #output1 = self.fc1(nz1)
        #output1 = self.relu(self.bn(output1.view(-1, 512, 4, 4)))
        output1 = self.main(nz1)

        #output2 = self.fc1(nz2)
        #output2 = self.relu(self.bn(output2.view(-1, 512, 4, 4)))
        output2 = self.main(nz2)

        return output1, output2

#
class D(nn.Module):
    def __init__(self, input_nc, ndf, k):
        super(D, self).__init__()
        self.main = nn.Sequential(
            # state 64 x 64
            nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
            # state 4 x 4
        )
        self.deconv5 = nn.Sequential(
            nn.Conv2d(ndf*8*k*2, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        output1 = self.main(input1)
        output2 = self.main(input2)
        output = self.deconv5(torch.cat([output1, output2], 1))

        output = self.fc1(output.view(-1, 2048))

        return output.view(-1, 1).squeeze(1)



# stack the input
class _netD(nn.Module):
    def __init__(self, input_nc, ndf, k):
        super(_netD, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(input_nc*2, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)








