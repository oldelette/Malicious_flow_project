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

def extract_feature(full_filename):

	pcap = rdpcap(full_filename)
	
	try:
		chack_ip_layer = pcap[0][IP]
		chack_ip_layer = pcap[-1][IP]
	except:
		return None
		
	if os.path.isfile('data.json'):
		os.remove('data.json')
		
	os.system(joy_path + " bidir=1 dist=1 entropy=1 tls=1 hd=1 " + full_filename + " | gunzip >> data.json")
	
	with open('data.json','r') as file:

		for i, line in enumerate(file.readlines()):	
			dic = json.loads(line)
			if i == 1:
				protocol  = dic["pr"]					
				dest_port = dic["dp"]
				
				if "bytes_out" in dic:
					bytes_out = dic["bytes_out"]
				else:
					bytes_out = 0
					
				if "num_pkts_out" in dic:
					num_pkts_out = dic["num_pkts_out"]
				else:
					num_pkts_out = 0
				
				if "bytes_in" in dic:
					bytes_in = dic["bytes_in"]
				else:
					bytes_in = 0
				
				if "num_pkts_in" in dic:
					num_pkts_in = dic["num_pkts_in"]
				else:
					num_pkts_in = 0						
				
				duration = dic["time_end"] - dic["time_start"]
				
				if ("ip" in dic) and ("out" in dic["ip"]):
					ip_out_ttl = dic["ip"]["out"]["ttl"]
				else:
					ip_out_ttl = 0
					
				if ("ip" in dic) and ("in" in dic["ip"]):
					ip_in_ttl = dic["ip"]["in"]["ttl"]
				else:
					ip_in_ttl = 0				
				
				bytes_dist = dic["byte_dist"]
					
	
	all_feature = [protocol, dest_port, bytes_out, num_pkts_out, bytes_in, num_pkts_in, duration, ip_out_ttl, ip_in_ttl]
	
	for byte_dist in bytes_dist:
		all_feature.append(byte_dist)
	
	if os.path.isfile('data.json'):
		os.remove('data.json')
		
	return all_feature # 265 features

def main():
	if os.path.isdir(csv_dir) == False:
		os.mkdir(csv_dir)
	
	if os.path.isdir(Weight_dir) == False:
		os.mkdir(Weight_dir)
	
	if os.path.isdir(result_dir) == False:
		os.mkdir(result_dir)
	
	
	feature_names = ["protocol", "dest_port", "bytes_out", "num_pkts_out", "bytes_in", "num_pkts_in",
				 "duration", "ip_out_ttl", "ip_in_ttl"]
				 
	for i in range(256):
		feature_names.append("byte_" + str(i))	

	feature_names.append("class")
	
	# generate train dataset
	train_data = []
	for dir_1 in class_dir:
		filenames = os.listdir(Pcap_dir+"train/"+dir_1)
		for filename in filenames:
			full_filename = Pcap_dir+"train/"+dir_1+filename			
			print(full_filename)
	
			row   = extract_feature(full_filename)
		
			if row == None:
				continue
			
			label = dir_1[:-1]
			row.append(label)
	
			train_data.append(row)
		
	train_df = pd.DataFrame(train_data, columns = feature_names)
	train_df.to_csv(csv_dir+train_joy_csv_filename, sep='\t')

	# generate test dataset
	test_data = []
	for dir_1 in class_dir:
		filenames = os.listdir(Pcap_dir+"test/"+dir_1)
		for filename in filenames:
			full_filename = Pcap_dir+"test/"+dir_1+filename
			print(full_filename)
	
			row   = extract_feature(full_filename)
		
			if row == None:
				continue
			
			label = dir_1[:-1]
			row.append(label)
	
			test_data.append(row)
		
	test_df = pd.DataFrame(test_data, columns = feature_names)
	test_df.to_csv(csv_dir+test_joy_csv_filename, sep='\t')

if __name__ == '__main__':
    main()
