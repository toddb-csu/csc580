# Todd Bartoszkiewicz
# CSC580: Applying Machine Learning and Neural Networks - Capstone
# Module 1: Critical Thinking Assignment
# Option #1
# For this assignment, you will write Python code to detect faces in an image. Use the following Python code as a
# starter for your program.
#
# Supply your own image file that contains one or more faces to identify. An example output for your program should be
# something like the following picture with red boxes (instead of green) drawn around each individual’s face:
#
# People walking with green boxes around their faces showing facial recognition
#
# import PIL.ImageDraw
# import face_recognition
# # Load the jpg file into a numpy array
# # Find all the faces in the image # Use the following Python pseudocode as guidance for your solution.
# numberOfFaces = len(faceLocations)print("Found {} face(s) in this picture.".format(numberOfFaces))
# Load the image into a Python Image Library object so that you can draw on top of it and display it
# pilImage = PIL.Image.fromarray(image)
# for faceLocation in faceLocations:
# Print the location of each face in this image. Each face is a list of co-ordinates in (top, right, bottom,
#  left) order.
#  print("A face is located at pixel location Top: {}, Left {},Bottom: {}, Right: {}".format(top, left, bottom, right))
#  # Draw a box around the face
#  drawHandle = PIL.ImageDraw.Draw(pilImage)
#  drawHandle.rectangle([left, top, right, bottom], outline="red")
# Display the image on screenpilImage.show()
# Develop the remaining code in the section specific to face detection.
# Because most human faces have roughly the same structure, the pre-trained face detection model will work well for
# almost any image. There's no need to train a new one from scratch. Use PIL, which is the Python Image Library.
#
# Submit your image file input and completed Python source as a zip file named:
# CSC580_CTA_1_1_last_name_first_name.zip.


import PIL.Image
import PIL.ImageDraw
import face_recognition


if __name__ == '__main__':
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file("download.jpg")
    # Find all the faces in teh image
    faceLocations = face_recognition.face_locations(image, number_of_times_to_upsample=2)

    numberOfFaces = len(faceLocations)
    print("Found {} face(s) in this picture.".format(numberOfFaces))

    # Load the image into a Python Image Library object so that you can draw on top of it and display it
    pilImage = PIL.Image.fromarray(image)

    # Loop through each face found to get coordinates of each face location and draw boxes around each face
    for faceLocation in faceLocations:
        # Print the location of each face in this image.
        top, right, bottom, left = faceLocation
        print("A face is located at pixel location Top: {}, Left {},Bottom: {}, Right: {}".format(top, left, bottom, right))

        # Draw a box around the face
        drawHandle = PIL.ImageDraw.Draw(pilImage)
        drawHandle.rectangle([left, top, right, bottom], outline="red", width=2)

    # Display the final image with red boxes around each face
    pilImage.show()

    # Save the new image file
    pilImage.save("detected_faces.jpg")
