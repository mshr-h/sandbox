import cv2
import argparse

parser = argparse.ArgumentParser(description="AKAZE algorithm example")
parser.add_argument("--input1", help="Path to the first input image.")
parser.add_argument("--input2", help="Path to the second input image.")

args = parser.parse_args()

img1 = cv2.imread(args.input1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(args.input2, cv2.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    print("Couldn't open the input image")
    exit(1)

akaze = cv2.AKAZE_create()
kpts1, desc1 = akaze.detectAndCompute(img1, None)
kpts2, desc2 = akaze.detectAndCompute(img2, None)

matcher = cv2.DescriptorMatcher_create("BruteForce")
matches = matcher.match(desc1, desc2)
matcher = sorted(matches, key=lambda x: x.distance)

img3 = cv2.drawMatches(img1, kpts1, img2, kpts2, matches[:10], None)

cv2.imshow("match result", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

