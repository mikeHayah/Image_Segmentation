#import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from PIL import Image
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import torch.nn as nn
import torch.nn.functional as F



def prepare_plot(origImage, origMask, predMask):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(100, 100))
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	# set the layout of the figure and display it
	figure.tight_layout()
	figure.show()
	
	

def make_predictions(state_dict_path, path):
	
	# create the output path
	output_path = path+'_downup/'
	if not os.path.exists(output_path):
		os.makedirs(output_path)
		 
	# Load the state dictionary
	state_dict = torch.load(state_dict_path)
	
	# Initialize the model
	model = U_Net(img_ch=1,output_ch=1)

	# Load the state dictionary into the model
	model.load_state_dict(state_dict)

	# Move the model to the desired device
	model = model.to('cpu')

	# set model to evaluation mode
	model.eval()
	
	
	# get images
	paths = list(map(lambda x: os.path.join(path, x), os.listdir(path)))
    

	# turn off gradient tracking
	with torch.no_grad():
		for image_path in paths:
			if image_path.endswith('.tif'):
				image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
				#orig = image.copy()
			
				image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to('cpu').float()
				
				premask, checkbtlnek =  model(image.to('cpu'))
				predMask = torch.sigmoid(premask)
				
				
				output = predMask.squeeze(0)
				output = output.squeeze(0)
				output = output - output.min()
				output = output / output.max()
				output = ((output>0.99) * 255).byte()  
				output_np = output.cpu().numpy()
				output_image = Image.fromarray(output_np)
				output_image.save(os.path.join(output_path,image_path.split('\\')[-1]))




				# # check changes via network
				# x, x1, x2, x3, x4, x5, d1, d2, d3, d4, d5 = model(image.to('cpu'))
				# output = [x, x1, x2, x3, x4, x5, d1, d2, d3, d4, d5]
				# for o in output:
				# 	conv = nn.Conv2d(1024, 1, kernel_size=1)
				# 	image_btlnek = conv(o)
				# 	btlnk_probs = F.sigmoid(image_btlnek)
				# 	stretching = nn.Upsample(scale_factor=16)
				# 	img_probs_stretched = stretching(btlnk_probs)
				# 	img_probs_stretched = img_probs_stretched.squeeze(0)
				# 	img_probs_stretched = img_probs_stretched.squeeze(0)
				# 	img_probs_stretched = img_probs_stretched - img_probs_stretched.min()
				# 	img_probs_stretched = img_probs_stretched / img_probs_stretched.max()
				# 	img_probs_stretched = ((img_probs_stretched>0.5) * 255).byte() 
				# 	img_probs_stretched = img_probs_stretched.cpu().numpy()
				# 	img_probs_stretched = Image.fromarray(img_probs_stretched)
				# 	img_probs_stretched.save(os.path.join(output_path,image_path.split('\\')[-1]+str(o)))


				
				#plt.imshow(mask_img)
				#predMask = predMask.astype(np.uint8)
				#plt.savefig("./dataset/prediction/try4.tif")
				# prepare a plot for visualization
				#prepare_plot(orig, gtMask, predMask)
		



