
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable, grad
from torch.nn.init import kaiming_normal, kaiming_uniform

from model.models import G, D, _netD
from data.dataset import aligned_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--flip', type=int, default=1, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--lamb', default=10, help='lamb for DRAGAN training')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu_ids', default=1, help='gpu ids: e.g. 0  0,1,2, 0,2.')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='results', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')


args = parser.parse_args()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    torch.cuda.set_device(args.gpu_ids)

cudnn.benchmark = True

# scale into (-1, 1)
# paired data loader
dataset = aligned_data(args.dataPath,args.imageSize, args.imageSize, args.flip)
train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=args.batchSize,
                                           shuffle=True,
                                           num_workers=2)
#normal data loader
'''
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
train_dataset = dset.ImageFolder(
    args.dataPath,
    transforms.Compose([
        transforms.Scale(64),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=2
)
'''

nz = int(args.nz)
ngf = int(args.ngf)
ndf = int(args.ndf)
nc = 3
#stack channels
k = 1
d_i = 50

#weights init method
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        #kaiming_uniform(m.weight.data, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#cunpute gradient penalty
def calc_gradient_penalty(netD, inputv1, inputv2):

    alpha1 = torch.rand(batch_size, 1)
    alpha1 = alpha1.expand(batch_size, inputv1.cpu().nelement() / batch_size).contiguous().view(input1.size())
    x_hat1 = Variable(alpha1 * inputv1.cpu().data + (1 - alpha1) * (
        inputv1.cpu().data + 0.5 * inputv1.cpu().data.std() * torch.rand(inputv1.size())), requires_grad=True).cuda()

    alpha2 = torch.rand(batch_size, 1)
    alpha2 = alpha2.expand(batch_size, inputv2.cpu().nelement() / batch_size).contiguous().view(input2.size())
    x_hat2 = Variable(alpha2 * inputv2.cpu().data + (1 - alpha2) * (
        inputv2.cpu().data + 0.5 * inputv2.cpu().data.std() * torch.rand(inputv2.size())),
                      requires_grad=True).cuda()

    pred_hat = netD(x_hat1, x_hat2)
    gradients1 = grad(outputs=pred_hat, inputs=x_hat1, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients2 = grad(outputs=pred_hat, inputs=x_hat2, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                      create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = lamb * ((gradients1.norm(2, dim=1) - 1) ** 2).mean() + lamb * (
    (gradients2.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


netG = G(nz, nc, ngf)
#netD = _neD(nc, ndf, k)
netD = D(nc, ndf, k)

print(netG)
print(netD)

netG.apply(weights_init)
netD.apply(weights_init)

input1 = torch.FloatTensor(args.batchSize, nc, args.imageSize, args.imageSize)
input2 = torch.FloatTensor(args.batchSize, nc, args.imageSize, args.imageSize)
z_i = torch.FloatTensor(args.batchSize, d_i, 1, 1)
z_o1 = torch.FloatTensor(args.batchSize, nz-d_i, 1, 1)
z_o2 = torch.FloatTensor(args.batchSize, nz-d_i, 1, 1)

fixed_z_i = torch.FloatTensor(args.batchSize,d_i, 1, 1).uniform_(-1, 1)

fixed_zo1 = torch.FloatTensor(1, nz-d_i, 1, 1).uniform_(-1, 1)
fixed_zo2 = torch.FloatTensor(1, nz-d_i, 1, 1).uniform_(-1, 1)
fixed_zo3 = torch.FloatTensor(1, nz-d_i, 1, 1).uniform_(-1, 1)
fixed_zo4 = torch.FloatTensor(1, nz-d_i, 1, 1).uniform_(-1, 1)

fixed_z_o1 = torch.FloatTensor(args.batchSize,nz-d_i, 1, 1)
fixed_z_o2 = torch.FloatTensor(args.batchSize,nz-d_i, 1, 1)
fixed_z_o3 = torch.FloatTensor(args.batchSize,nz-d_i, 1, 1)
fixed_z_o4 = torch.FloatTensor(args.batchSize,nz-d_i, 1, 1)

fixed_z_o1.copy_(fixed_zo1)
fixed_z_o2.copy_(fixed_zo2)
fixed_z_o3.copy_(fixed_zo3)
fixed_z_o4.copy_(fixed_zo4)

fixed_z1 = Variable(torch.cat([fixed_z_i, fixed_z_o1], 1))
fixed_z2 = Variable(torch.cat([fixed_z_i, fixed_z_o2], 1))
fixed_z3 = Variable(torch.cat([fixed_z_i, fixed_z_o3], 1))
fixed_z4 = Variable(torch.cat([fixed_z_i, fixed_z_o4], 1))

label = torch.FloatTensor(args.batchSize)
real_label = 1
fake_label = 0

criterion = nn.BCELoss()
#MSEloss for LSGAN
#criterion = nn.MSELoss()

if args.cuda:
    netG.cuda()
    netD.cuda()
    criterion.cuda()
    input1, input2 = input1.cuda(), input2.cuda()
    z_i, z_o1, z_o2 = z_i.cuda(), z_o1.cuda(), z_o2.cuda()
    fixed_z1, fixed_z2, fixed_z3, fixed_z4 = fixed_z1.cuda(), fixed_z2.cuda(), fixed_z3.cuda(), fixed_z4.cuda()
    label = label.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

lamb = 10

for epoch in range(args.niter):
    for i, data in enumerate(train_loader, 0):

        netD.zero_grad()
        image1 = data[0]
        image2 = data[1]
        #print(image1.size())
        batch_size = image1.size(0)

        if args.cuda:
            image1 = image1.cuda()
            image2 = image2.cuda()

        input1.resize_as_(image1).copy_(image1)
        input2.resize_as_(image1).copy_(image2)
        label.resize_(batch_size).fill_(real_label)
        inputv1 = Variable(input1)
        inputv2 = Variable(input2)
        labelv = Variable(label)

        output = netD(inputv1, inputv2)
        #print(labelv, output)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        z_i.resize_(batch_size, d_i, 1, 1).uniform_(-1, 1)
        z_o1.resize_(batch_size, nz-d_i, 1, 1).uniform_(-1, 1)
        z_o2.resize_(batch_size, nz-d_i, 1, 1).uniform_(-1, 1)

        z_1 = Variable(torch.cat([z_i, z_o1], 1))
        z_2 = Variable(torch.cat([z_i, z_o2], 1))

        G1, G2 = netG(z_1, z_2)
        fakeAB = torch.cat([G1, G2], 1)
        labelv = Variable(label.fill_(fake_label))
        output = netD(G1.detach(), G2.detach())

        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        #gradient nrom
        #batch_size = imgA.size(0)

        errD = errD_real + errD_fake
        optimizerD.step()

        #Update G
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))
        output = netD(G1, G2)

        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()

        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, args.niter, i, len(train_loader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        #save results every 100 iters
        if i % 100 == 0:
            vutils.save_image(image1,
                    '%s/real_samples1.png' % args.outf,
                    normalize=True)
            vutils.save_image(image2,
                              '%s/real_samples2.png' % args.outf,
                              normalize=True)
            #inference
            G1, G2 = netG(fixed_z1, fixed_z2)
            G3, G4 = netG(fixed_z3, fixed_z4)
            G_all = torch.cat([G1, G2, G3, G4], 0)
            vutils.save_image(G_all.data,
                    '%s/G_samples_epoch_%03d.png' % (args.outf, epoch),
                    normalize=True, nrow=16)
            '''
            vutils.save_image(G2.data,
                              '%s/G2_samples_epoch_%03d.png' % (args.outf, epoch),
                              normalize=True)
            
            vutils.save_image(G3.data,
                              '%s/G3_samples_epoch_%03d.png' % (args.outf, epoch),
                              normalize=True)

            vutils.save_image(G4.data,
                              '%s/G4_samples_epoch_%03d.png' % (args.outf, epoch),
                              normalize=True)
            '''
    #save G and D model every 20 epochs
    if epoch % 20 ==0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))






























