import torch
import torch.nn as nn
import numpy as np
import glob
import cv2

# Define the dehaze network as a subclass of torch.nn.Module
class dehaze_net(nn.Module):

    def __init__(self):
        super(dehaze_net, self).__init__()

        # Define the layers of the network
        self.relu = nn.ReLU(inplace=True) 
        self.e_conv1 = nn.Conv2d(3,3,1,1,0,bias=True)  
        self.e_conv2 = nn.Conv2d(3,3,3,1,1,bias=True)  
        self.e_conv3 = nn.Conv2d(6,3,5,1,2,bias=True)  
        self.e_conv4 = nn.Conv2d(6,3,7,1,3,bias=True)  
        self.e_conv5 = nn.Conv2d(12,3,3,1,1,bias=True)  
        
    def forward(self, x):
        source = []
        source.append(x)

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))

        concat1 = torch.cat((x1,x2), 1)
        x3 = self.relu(self.e_conv3(concat1))

        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.e_conv4(concat2))

        concat3 = torch.cat((x1,x2,x3,x4),1)
        x5 = self.relu(self.e_conv5(concat3))

        clean_image = self.relu((x5 * x) - x5 + 1) 
        
        return clean_image

# Create an instance of the dehaze network
model = dehaze_net()

# Load the pre-trained weights into the model
model.load_state_dict(torch.load('snapshots/dehazer.pth',map_location=torch.device('cpu')))

# Set the device to CPU
device = torch.device('cpu')
model.to(device)

def dehaze_image(image_path):
    image=cv2.imread(image_path)
    frame=image
    # Convert the frame to RGB color space
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a PyTorch tensor and move it to the CPU
    frame = torch.from_numpy(frame.transpose((2, 0, 1))).float().unsqueeze(0) / 255.0
    frame = frame.to(device)

    # Apply the dehazing model to the frame and move the result back to the CPU
    with torch.no_grad():
        dehazed_frame = model(frame).squeeze().cpu().numpy()

    # Convert the dehazed frame to BGR color space
    dehazed_frame = (dehazed_frame * 255).clip(0, 255).transpose((1, 2, 0)).astype(np.uint8)
    dehazed_frame = cv2.cvtColor(dehazed_frame, cv2.COLOR_RGB2BGR)

    cv2.imwrite("results/" + image_path.split("\\")[-1], cv2.hconcat([image,dehazed_frame]))
	

if __name__ == '__main__':

	test_list = glob.glob("test_images/*")

	for image in test_list:

		dehaze_image(image)
		print(image, "done!")
