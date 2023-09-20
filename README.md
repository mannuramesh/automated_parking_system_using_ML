# automated_parking_system_using_ML

The automated parking system using Machine Learning, uses OpenCv for detection, tessaract application for OCR (optical Character Reader), and pyQt5 for application UI interface.

Its an amazing applicataion, which reads the character on number plate, and print out the results, there are three way it is done, 

-by normal detection (detect.py), which loads image specified in code and shows the OCR output.

-by batch detection (batch_detect.py), which reads multiple images from the images folder, the OCR reads the character and saves the output in a vehicle_numbers.txt file, we have achieved 88% of accuracy, out 6 images, 5 was correct and the wrong image has one character missing which means, it almost captures the character and detects the.

-the another way is, by ui application(qt5_select.py), which enables user to select specific file each time and the OCR reads the input and shows the OCR image and the with text character detected, it almost detect 80% of accuracy, because it detects the license panels but could not read OCR character properly, maybe the performance of opencv degrades with heavy UI application


OpenCV is a good library for real world object detection, but its not efficient, it needs a better library like tensorflow, YOLO libraries and heavily trained datasets for achieving higher accuracy upto 98%
