# Import required libraries
import cv2 as cv
import threading
import queue
import torch
import numpy as np
import torch.nn as nn
import time
import argparse

camera=0 # 0 means default camera

# Create the parser
parser = argparse.ArgumentParser()

# Add the argument
parser.add_argument("-r","--resolution",required=True, help="give the resolution of image, More the resolution, more the time took, value in percentage")
# Parse the arguments
args = parser.parse_args()

# Get the argument value
res = int(args.resolution)

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


# Define a function to dehaze a video stream in a separate thread
def thread_dehaze(stop_event, result_queue):
    # Initialize the video capture object and check if it's successfully opened
    cap = cv.VideoCapture(camera)# Load the video capture, change -1 to 0 to use the code in windows !important
    frame=cap.read()
    assert cap.isOpened() and frame[0] == True
    Ratio = frame[1].shape[:2][::-1]  # Aspect ratio of camera (not in simplified form), format=[width,height]

    while not stop_event.is_set():
        # Read the video frame
        (success, frame) = cap.read()

        # If frame is not read properly, break and go for another frame
        if not success:
            break

        # Start timer to measure performance
        s = time.perf_counter()

        # Resize the frame with the set resolution and aspect ratio
        frame = cv.resize(frame, (int(res*Ratio[0]/100), int(res*Ratio[1]/100)))# Resize image for fast dehazing
        orig=frame

        # Convert the frame to RGB color space
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Convert the frame to a PyTorch tensor and move it to the CPU
        frame = torch.from_numpy(frame.transpose((2, 0, 1))).float().unsqueeze(0) / 255.0
        frame = frame.to(device)

        # Apply the dehazing model to the frame and move the result back to the CPU
        with torch.no_grad():
            dehazed_frame = model(frame).squeeze().cpu().numpy()

        # Convert the dehazed frame to BGR color space
        dehazed_frame = (dehazed_frame * 255).clip(0, 255).transpose((1, 2, 0)).astype(np.uint8)
        dehazed_frame = cv.cvtColor(dehazed_frame, cv.COLOR_RGB2BGR)

        # Put the dehazed frame into the result queue
        result_queue.put([dehazed_frame,orig])
        print("Frames Per Second(FPS): {}".format(1/(time.perf_counter()-s)))#FramesPerSecond

    # Release the video capture object
    cap.release()

def update_value(x):
    global res
    res = x
# Define the callback function
def on_mouse(event, x, y, flags, param):
    global shutdown

    if event == cv.EVENT_LBUTTONDOWN:
        # Store the current time when the left mouse button is pressed down
        param['start_time'] = cv.getTickCount()
    elif event == cv.EVENT_LBUTTONUP:
        # Calculate the duration of the left mouse button press
        end_time = cv.getTickCount()
        duration = (end_time - param['start_time']) / cv.getTickFrequency()

        # If the duration is longer than 3 seconds, shutdown
        if duration > 3:
            print("Will NOT shutdown,but end program since this is a test,not Pi(even if its pi,this wont shut down as its test code)")
            cv.destroyAllWindows()
            shutdown=1
            stop_event.set()
        else:
            print("Long touch to exit")


if __name__ == "__main__":
    # Create an event object to signal the thread to stop and a queue to hold the most recently dehazed image
    stop_event = threading.Event()
    result_queue = queue.Queue(maxsize=1)  # Thread queue contains only one image, that is the most recently dehazed

    # Start the dehazing thread
    thread = threading.Thread(target=thread_dehaze, args=(stop_event, result_queue))
    thread.start()

    shutdown=0 #If button has been ever pressed
    button_down = False # Button state
    
    # Create a named window and set it to full screen
    cv.namedWindow("Original",cv.WINDOW_GUI_EXPANDED)
    cv.namedWindow("Dehazed", cv.WINDOW_GUI_EXPANDED)

    cv.resizeWindow("Original", 400, 300)
    cv.resizeWindow("Dehazed", 400, 300)

    while True:
        while cv.getWindowProperty('Original', cv.WND_PROP_VISIBLE) >= 1 and cv.getWindowProperty('Dehazed', cv.WND_PROP_VISIBLE) >= 1:

            # Handle new result, if any
            try:
                # Get the most recently dehazed frame from the result queue
                result_frame = result_queue.get_nowait()
                cv.imshow("Original", result_frame[1])# Show the most recently dehazed frame's original version in the window
                cv.imshow("Dehazed", result_frame[0])# Show the most recently dehazed frame in the window

                cv.setMouseCallback("Original", on_mouse,{'start_time': 0})#Add a callback event
                cv.setMouseCallback("Dehazed", on_mouse,{'start_time': 0})#Add a callback event

                # Mark the task as done so that the queue frees up memory
                result_queue.task_done()
            
            except queue.Empty:
                pass

            # Process GUI events
            key = cv.waitKey(1)
            # If Enter key is pressed, break the loop and stop the dehazing thread
            if key == 13 or shutdown==1:#When shutdown key or enter pressed,break the loop and close the program
                break
        if key == 13 or shutdown==1:#When shutdown key or enter pressed,break the loop and close the program
            break
        else:
            print("Long touch any of the two window to exit")
            cv.namedWindow("Original", cv.WINDOW_GUI_EXPANDED)
            cv.namedWindow("Dehazed", cv.WINDOW_GUI_EXPANDED)

    # Signal the dehazing thread to stop and wait for it to join
    stop_event.set()
    thread.join()