import cv2 as cv
import threading
import queue
import torch
import numpy as np
import torch.nn as nn
import math
import time


class dehaze_net(nn.Module):

    def __init__(self):
        super(dehaze_net, self).__init__()

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
model = dehaze_net()

model.load_state_dict(torch.load('snapshots/dehazer.pth',map_location=torch.device('cpu')))

device = torch.device('cpu')
model.to(device)

def worker_function(stop_event, result_queue):

    res=60 #more the resolution,more the time
    Ratio=[4,3] #Aspect ratio of camera, format=[width,height]

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, res*Ratio[0])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, res*Ratio[1])
    assert cap.isOpened()

    while not stop_event.is_set():
        (success, frame) = cap.read()
        if not success: break

        # do your inference here
        # print(frame.shape) #(480, 640, 3)
        s=time.perf_counter()
        # frame = cv.resize(frame, (320, 240))
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame.transpose((2, 0, 1))).float().unsqueeze(0) / 255.0
        frame = frame.to(device)

        with torch.no_grad():
            dehazed_frame = model(frame).squeeze().cpu().numpy()

        dehazed_frame = (dehazed_frame * 255).clip(0, 255).transpose((1, 2, 0)).astype(np.uint8)
        dehazed_frame = cv.cvtColor(dehazed_frame, cv.COLOR_RGB2BGR)
        print(time.perf_counter()-s)

        result_queue.put(dehazed_frame)

    cap.release()


if __name__ == "__main__":
    stop_event = threading.Event()
    result_queue = queue.Queue(maxsize=1)
    worker_thread = threading.Thread(
        target=worker_function, args=(stop_event, result_queue))
    worker_thread.start()

    cv.namedWindow("window", cv.WINDOW_NORMAL)

    while True:
        # handle new result, if any
        try:
            result_frame = result_queue.get_nowait()
            cv.imshow("window", result_frame)
            result_queue.task_done()
        except queue.Empty:
            pass

        # GUI event processing
        key = cv.waitKey(1)
        if key in (13, 27): # Enter, Escape
            break
    
    stop_event.set()
    worker_thread.join()
