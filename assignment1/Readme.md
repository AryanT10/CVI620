## Part I: A photo booth application

you need a webcam, or a digital camera connected and installed on your machine.
A code to capture and show the video stream from your webcam (or camera).

## Part II: Image Arithmetic!

### A.	Brightness and Contrast: 

  i). Open a color image and display.

  ii). Increased the brightness by adding a constant (e.g., 150) to all color channels of the image. Display in a separate window.
    
  iii).	Change the contrast by multiplying the image by a constant (e.g., 0.5). Display in a separate window. ![image](https://github.com/user-attachments/assets/5cf32df3-4475-4f5b-a19a-ff95f62c5231)

### B. Linear blend: 

  i).	Open a second images and display. Resize the second image to match the first, if needed.

  ii). Ask the user for a number (alpha) between 0 and 1. (I ENTERED 0.5)

  iii). Implemented a linear blend of the two images: 
                                      blend = (1 - alpha) * img1 + alpha * img2;


## Part III: A Drawing Application

  i). Created a program to draw green rectangles on a image with thickness is 4. 

  ii). Changed thickness to -1. What do you notice? Explain.
      
    •	Before (thickness 4) only the borders of rectangles were drawn, the inside of the rectangle was transparent, and the original image was visible through the rectangle
      
    •	After (thickness to -1) the entire area inside the rectangle is completely filled with green colour and the original image content in the rectangle is covered and no longer visible.

iii). Created a program to put Text On the Rectangle in the Image


