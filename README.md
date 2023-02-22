# ATL Marathon 2023
This project is private, but members of this organisation can view, edit, merge.

This is the most simplified version of https://github.com/MayankSingal/PyTorch-Image-Dehazing.

The project supports video dehazing, live on the Raspberry Pi 400.

# Run:
Make sure that the shell environment is in the directory with `main.py` and `dehazer.pth`

```sh
cd /path/to/python/file
python main.py
```
To ensure the file opens at startup of desktop environment, place `startup.desktop` in `/etc/xdg/autostart/`.

Add the location of `startup.sh` in `startup.desktop`.
```.desktop
Exec=/path/to/startup.sh
```
Make sure to set `startup.sh`'s permissions to `+x` so it can be executed. `startup.sh` is recommended to be placed in `/path/to/python/file`

In `startup.sh`, make sure to change the directory path.
```sh
#!/bin/sh

cd /path/to/python/file
python3 main.py
```

# Train, test.:
See original repository. 

Train there, import Model here.
