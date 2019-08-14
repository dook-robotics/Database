## reducing_big_images.py ##
## Augment dataset with crops of images of a set size ##
## while preserving the xml data                      ##
#
# Authors:
#   Mikian Musser - https://github.com/mm909
#   Eric Becerril-Blas - https://github.com/lordbecerril
#   Zoyla O - https://github.com/ZoylaO
#   Austin Janushan - https://github.com/Janushan-Austin
#   Giovanny Vazquez - https://github.com/giovannyVazquez
#
# Organization:
#   Dook Robotics - https://github.com/dook-robotics
#
# Usage:
#	python reducing_big_images.py
#
# Todo:
#	Take directories in as arguments
#


import os
from impy.ObjectDetectionDataset import ObjectDetectionDataset

def main():
	# Define the path to images and annotations
	images_path = os.path.join(os.getcwd(), "images", "train", "image")
	annotations_path = os.path.join(os.getcwd(), "images", "train", "xml")
	# Define the name of the dataset
	dbName = "CardsDataset"
	# Create an object of ObjectDetectionDataset
	obda = ObjectDetectionDataset(imagesDirectory=images_path, annotationsDirectory=annotations_path, databaseName=dbName)
	# Reduce the dataset to smaller Rois of smaller ROIs of shape 1032x1032.
	# offset=[1032, 1032]
	offset=[300, 300]
	images_output_path = os.path.join(os.getcwd(), "reducedImages", "train")
	annotations_output_path = os.path.join(os.getcwd(), "reducedImages", "train")
	obda.reduceDatasetByRois(offset = offset, outputImageDirectory = images_output_path, outputAnnotationDirectory = annotations_output_path)

if __name__ == "__main__":
	main()
