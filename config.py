# Set the resolution and aspect ratio of the camera
res = 19    # More the resolution, more the time took, value in percentage,
            # this is only default value set (with respect to the project's 1280x720, so the VALUE MAY VARY FOR EACH CAMERA),
            # it can be adjusted in GUI with slider.
             
camera = 0 # 0 means default camera

swipe_threshold = 50 # Percent,for a swipe to be detected
shutdown_threshold = 50 #Time in seconds for long press to be detected

debug_mode = False
shutdown_command = "sudo shutdown -h now"
dehazemodel = 'dehazer.pth'
fps = 5