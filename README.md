# ATL Marathon 2023
This is our submission to the ATL Marathon 2023.

The project supports video dehazing, live on the Raspberry Pi 400.

# Run:
* Make sure that the shell environment is in the directory with `main.py`, `button.png` and `dehazer.pth`
```sh
cd /path/to/python/file
python main.py
```

* To ensure the file opens at startup of desktop environment, place `to add in other folders/startup.desktop` in `/etc/xdg/autostart/`.

* Add the location of `startup.sh` in `To add in other folders/startup.desktop`.
```.desktop
Exec=/path/to/startup.sh
```

* Make sure to set `startup.sh`'s permissions to `rwxr-xr-x` so it can be executed. 
```sh
chmod rwxr-xr-x /path/to/startup.sh
```

* `startup.sh` is recommended to be placed in `/path/to/python/file`

* In `startup.sh`, make sure to change the directory path.
```sh
#!/bin/sh

cd /path/to/python/file
python3 main.py
```

* `To add in other folders/road edit 1.png` has been set as the background image of the desktop environnment.

# Train, test.:
See [original repository](https://github.com/MayankSingal/PyTorch-Image-Dehazing) for information on training.

Train there, import the Model (`dehazer.pth`) here.
