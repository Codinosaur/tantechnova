# TestImageDehaze.py
## Description
Tests a single image dehazing with same algorithm.

## Usage
```cmd
python TestImageDehaze.py
```
## Result
The code gets images in `test_images directory/`, stores dehazed image in `results/` directory.

# TestVideoDehazing.py
## Description
Tests video dehazing with same algorithm. Shows original, and dehazed live video in separate window for each.

## Usage
```cmd
python TestVideoDehazing.py -resolution 25
rem Below is alternative short version
python TestVideoDehazing.py -r 25 
```
## Result
The code gets images from video captured dehazes and show both the original, dehazed code for comparison, in two separate windows.
