import torch
import matplotlib.pyplot as plt

# NEEDS TO BE FIXED

def show_cam(model, inputs, predicted, targets, images, sample_fnames):
    model_wrapper = ModelWrapper(model)
    x, cams, apool = model_wrapper.forward(inputs)

    for i in range(len(cams)):
        plt.ion()
        fig = plt.figure(figsize=(100,100))
        ax = fig.add_subplot(1,2,1)


        im1 = plt.imshow(cams[i])
        plt.title("Pred: " +['non_metal', 'metal'][predicted[i]]+" True: " +['non_metal', 'metal'][targets[i]])
        fig.colorbar(im1, ax=ax)
        fig.add_subplot(1,2,2)
        plt.imshow(images[i,:,:,:].permute(1, 2, 0), cmap='gray')
        plt.title(sample_fnames[i])
        plt.waitforbuttonpress()
        plt.close()


class ModelWrapper:

    # image_channels, works with both 3=RGB and 1=grayscale
    def __init__(self, model):
        self.model = model
        #self.model.fc = nn.Linear(512,2)
        #self.model.sm = nn.Softmax(dim=1)


    def forward(self, x):
        # Fult hack som kanske löser problem vid 1 kanal
        #if len(x.shape) < 4:
            #x = x.expand(1, -1, -1, -1)
            #x = x.permute(1, 0, 2, 3)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        cams = []
        for i in range(4):
            #gmp = torch.max(x[i, :,:,:], 0).values
            #gap = torch.mean(x[i, :,:,:], 0)
            #cam = self.model.sm(gmp)
            #cams.append(cam.cpu())
            cam = torch.zeros((32,64, 2))
            for k in range(32):
                for n in range(64):
                    cam[k,n] = self.model.fc(x[i,:,k,n])
            cam = cam[:,:,1] - cam[:,:,0]
            cams.append(cam)

        #x = self.model.fc(x.cpu())

        x = self.model.avgpool(x)
        apool = x
        #x = x.reshape(x.shape[0], -1)
        #x = self.model.fc(x)

        return x, cams, apool
