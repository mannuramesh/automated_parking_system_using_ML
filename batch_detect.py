import numpy as np
import cv2
import imutils
import pytesseract
import os


def detect_license_plate(img_path):
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # Resize the image
    img = imutils.resize(img, width=500)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Detect edges using Canny
    edged = cv2.Canny(gray, 170, 200)

    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCnt = None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            break

    if NumberPlateCnt is None:  # <-- Here's the change
        return None # Return None if no plate was found

    # Mask the rest of the image apart from the number plate
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Extract text from the masked image using Tesseract
    config = ('-l eng --oem 1 --psm 3')
    text = str(pytesseract.image_to_string(new_image, config=config))
    return text


if __name__ == "__main__":
    directory = "images"  # The directory where images are stored
    output_file = "vehicle_numbers.txt"

    with open(output_file, 'w') as f:
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                full_path = os.path.join(directory, filename)
                license_number = detect_license_plate(full_path)
                if license_number:
                    f.write(f"{filename}: {license_number.strip()}\n")

    print(f"Vehicle numbers have been saved to {output_file}.")
