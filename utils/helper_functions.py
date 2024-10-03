
import dlib
import cv2
import matplotlib.pyplot as plt

def calculate_variance(path : str):
  """
  This function recognizes a face from an image and calculates
  the variance after using a Laplace transformation. 

  Args:
    path (str) : path to the image to be analyzed
  
  Returns:
    lapl_var (float) : variance after Laplace transformation
    dets (dlib.rectangles) : the detected face rectangles
  """
  # Load the image in gray scale for analysis purposes
  img = dlib.load_grayscale_image(path)

  # Initialize the face detector
  detector = dlib.get_frontal_face_detector()

  # Detect the faces in the image
  dets = detector(img)

  # If the number of faces is larger than 1, print error and exit the function
  if len(dets) > 1:
      print("Error: The number of faces recognized is larger than one. Please upload a picture where only one face can be recognized.")
      return None, None  # Exit the function early

  # Continue with further analysis if only one face is detected
  if len(dets) == 0:
      print("Error: No face detected. Please upload a picture with a recognizable face.")
      return None, None  # Exit if no face is detected

  for face in dets:
    # Get the coordinates of the bounding box containing the face
    x, y, w, h = face.left(), face.top(), face.width(), face.height()

    # Crop the face from the image
    face_region = img[y:y+h, x:x+w]

  # Remove noise by blurring with a Gaussian filter
  face_region = cv2.GaussianBlur(
      src=face_region,
      ksize=(3, 3), # kernel size
      sigmaX=0 # standard deviation in x direction
  )

  # Apply Laplace function
  laplacian = cv2.Laplacian(src=face_region,
                            ddepth=cv2.CV_16S,
                            ksize=3)
  
  # Obtain the mean value and the standard deviation of the img after Laplace transformation
  img_abs = cv2.convertScaleAbs(laplacian)
  lapl_mean, lapl_std = cv2.meanStdDev(img_abs)

  # Calculate the variance (measure for the sharpness of the image)
  lapl_var = lapl_std.item()**2

  return lapl_var, dets  # Return variance and face rectangles

def focus_measure_and_plot(path : str, treshold : float = 100.0):
  """
  This function plots a recognized face with the variance after a
  Laplace transformation.

  Args:
    path (str) : Path to the image to be analyzed.
    treshold (float) : Measure of sharpness. If this value lies below the
    variance of the image, then the image is sharp enough.
  """
  # Calculate the variance and detect faces
  lapl_var, dets = calculate_variance(path)

  # If dets is None, exit
  if dets is None:
      return

  # Load the image in RGB scale for visualization
  img = dlib.load_rgb_image(path)

  # Draw a green rectangle around the detected face
  for face in dets:
    cv2.rectangle(img=img, 
      pt1=(face.left(), face.top()), 
      pt2=(face.right(), face.bottom()), 
      color=(0, 255, 0), # (R, G, B)
      thickness=2)
    
  # If the variance of the image lies below the treshold, the color of the title will be green, otherwise red
  colour = 'g' if lapl_var > treshold else 'r'

  # Plot the color image with the variance
  plt.imshow(img)
  plt.title(f"Variance: {lapl_var:.2f}", color=colour)
  plt.axis(False) # Turn off the axis
  plt.show()

  # If the face is not sharp enough, print out a message asking the user to take upload an image where the face is in focus
  if lapl_var < treshold:
    print("The face cannot be recognized. Please upload an image where the face is in focus.")
