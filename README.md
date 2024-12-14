# Document Scanner and Image Rectification Script

## Overview
This Python script performs advanced image processing to detect, extract, and rectify documents from an image, transforming them into a clean, scanned-like document.

## Features
- Document detection using edge detection
- Contour identification
- Corner detection
- Perspective transformation (document rectification)
- Grayscale and thresholded output generation

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib

## Dependencies Installation
```bash
pip install opencv-python numpy matplotlib
```

## Usage
1. Place your input image (`matttest1.jpg`) in the same directory as the script
2. Run the script
3. The script will generate several output images:
   - Resized grayscale image
   - Canny edge detection result
   - Hough transform lines
   - Detected document corners
   - Rectified high-quality image
   - Rectified grayscale image
   - Rectified 'scanned' image with adaptive thresholding

## Script Workflow
1. Read and resize input image
2. Convert to grayscale
3. Apply Gaussian blur
4. Perform Canny edge detection
5. Find document contour
6. Detect document corners
7. Apply perspective transformation
8. Generate various output representations

## Output
- `resized_image.jpg`: Resized grayscale version of original image
- Multiple visualization plots showing intermediate processing steps
- Final rectified document images

## Customization
You can modify parameters like:
- `sigma` for Gaussian blur
- Canny edge detection thresholds
- Morphological operation kernel size

## Potential Applications
- Document digitization
- Paper scanning
- Image preprocessing for OCR
- Perspective correction for document images

## Troubleshooting
- Ensure input image is clear and document edges are visible
- Adjust blur and edge detection parameters if automatic detection fails
- Verify all required libraries are installed

## License
MIT License

Copyright (c) 2024 Nosherwan Babar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author
Nosherwan Babar
