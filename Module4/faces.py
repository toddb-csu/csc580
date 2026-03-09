# Todd Bartoszkiewicz
# CSC580: Applying Machine Learning and Neural Networks - Capstone
# Portfolio Project Milestone
# Option #1: Implementing Facial Recognition
# For this Portfolio Project Milestone, you will build on the programming requirements presented in Critical Thinking
# Assignment, Module 1, Option 1. In this Milestone, you will implement a Python program that uses facial recognition to
# determine if an individual face is present in a group of faces. For example, your program will use facial recognition
# to determine that the following individual:
#
# Man using a tablet.
# is not in the group of people shown here:
# Group of people standing in line.
#
# Hints for implementing such a program are found in the following video:
# Description: In the video, Deep Learning: Face Recognition
# https://www.linkedin.com/learning-login/share?account=2245842&forceAccount=false&redirect=https%3A%2F%2Fwww.linkedin.com%2Flearning%2Fdeep-learning-face-recognition%3Ftrk%3Dshare_ent_url%26shareId%3DhKXYVKTARP6xxKhdm9MkmA%253D%253D,
# you learn that face recognition is used for everything from automatically tagging pictures to unlocking cell phones.
# And with recent advancements in deep learning, the accuracy of face recognition has improved. In this course, learn
# how to develop a face recognition system that can detect faces in images, identify the faces, and even modify faces
# with "digital makeup" like you've experienced in popular mobile apps. Find out how to set up a development
# environment. Discover tools you can leverage for face recognition. See how a machine learning model can be trained to
# analyze images and identify facial landmarks. Learn the steps involved in coding facial feature detection,
# representing a face as a set of measurements, and encoding faces. Additionally, learn how to repurpose and adjust
# pre-existing systems. Submit your Python code and image data using a zip file entitled:
# CSC526_MidTermPortfolio _Option_1_last_name_first_name.zip


import face_recognition
import cv2
import sys
import numpy as np


if __name__ == '__main__':
    # Load the individual image
    user_image = face_recognition.load_image_file("man_holding_tablet.jpg")
    user_encodings = face_recognition.face_encodings(user_image)

    if len(user_encodings) == 0:
        print("No face detected in the image.")
    else:
        user_encoding = user_encodings[0]

        group_image = face_recognition.load_image_file("group_of_people.jpg")
        # face_locations = face_recognition.face_locations(group_image, model="cnn")
        face_locations = face_recognition.face_locations(group_image, number_of_times_to_upsample=2)
        group_encodings = face_recognition.face_encodings(group_image, face_locations)

        if len(group_encodings) == 0:
            print("No faces found in the group image.")
        else:
            matches = []
            is_found = False

            for index, face_encoding in enumerate(group_encodings):
                match_results = face_recognition.compare_faces([user_encoding], face_encoding, tolerance=0.6)
                distance = face_recognition.face_distance([user_encoding], face_encoding)[0]
                is_match = match_results[0]
                matches.append((is_match, distance))

                if is_match:
                    is_found = True
                    print(f"Match found. Distance = {distance:.4f}")
                else:
                    print(f"No match for face {index+1}. Distance = {distance:.4f}")

            if is_found:
                print("The individual IS present in the group photo.")
            else:
                print("The individual is NOT present in the group photo.")

            my_image = cv2.cvtColor(group_image, cv2.COLOR_RGB2BGR)

            for (top, right, bottom, left), (is_match, dist) in zip(face_locations, matches):
                color = (0, 255, 0) if is_match else (0, 0, 255)
                cv2.rectangle(my_image, (left, top), (right, bottom), color, 3)

                label = f"Match: {dist:.3f}" if is_match else f"No: {dist:.3f}"
                cv2.putText(my_image, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Face Recognition Result", my_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("results.jpg", my_image)
