import binascii
import os
import socket

# Third party library
import json
import numpy as np
from PIL import Image
from scapy.all import *
import pandas as pd

# Our library
from config import *
from utils_func import *

def plot_flow_packet_direction(full_filename):

	pcap = rdpcap(full_filename)
	
	src_ip = pcap[0][IP].src	
	
	packet_value_list = [0 for i in range(500)]
	bins = [i for i in range(500)]
	
	for i, pkt in enumerate(pcap):
		if i < 500:
			if src_ip == pkt[IP].src:
				packet_value_list[i] = len(pkt)
			else:
				packet_value_list[i] = len(pkt) * (-1)
	
	print(packet_value_list)
	
	filename = full_filename.split("/")[-1]
	fig = plt.figure(num=filename, figsize=(16, 9))		
	plt.xlabel(filename)
	plt.ylabel("packet number")
	plt.ylim(-1500, 1500)
	plt.bar(bins, packet_value_list, alpha=1, color="orange",  label="packet number") 

	#plt.hist([x, y], bins=bins, stacked=True)
	#plt.legend(loc="upper right")		
	filename = analysis_dir + "3_" + filename + '.png' 
	plt.savefig(filename)
		

def main():
	if os.path.isdir(analysis_dir) == False:
		os.mkdir(analysis_dir)
		
	for dir_1 in class_dir:
		filenames = os.listdir(Pcap_dir+"train/"+dir_1)
		for filename in filenames:
			full_filename = Pcap_dir+"train/"+dir_1+filename			
			print(full_filename)
			plot_flow_packet_direction(full_filename)
			break
			
			

if __name__ == '__main__':
    main()
