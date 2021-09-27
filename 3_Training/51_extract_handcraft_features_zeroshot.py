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

# Forward Inter Arrival Time, the time between two packets sent forward direction (mean, min, max, std)
def extract_fiat(pcap):
	time_interval_list = []
	first_time = True
	previous_timestamp = 0
	
	src_ip = pcap[0][IP].src
	
	
	for pkt in pcap:
		if first_time and (src_ip in pkt[IP].src):
			first_time = False
			previous_timestamp = pkt.time
			
		elif (src_ip in pkt[IP].src):
			now_timestamp = pkt.time
			timestamp_diff = now_timestamp - previous_timestamp
			time_interval_list.append(timestamp_diff)
			previous_timestamp = now_timestamp
	
	if len(time_interval_list) == 0:	 
		min_fiat = 0
		max_fiat = 0
		mean_fiat = 0
		std_fiat = 0
	else:		
		min_fiat = min(time_interval_list)
		max_fiat = max(time_interval_list)
		mean_fiat = np.mean(time_interval_list)
		std_fiat = np.std(time_interval_list)
	
	return min_fiat, max_fiat, mean_fiat, std_fiat
	

# Backward Inter Arrival Time, the time between two packets sent backwards (mean, min, max, std)	
def extract_biat(pcap):
	time_interval_list = []
	first_time = True
	previous_timestamp = 0
	
	src_ip = pcap[0][IP].src
	
	for pkt in pcap:
		if first_time and (src_ip in pkt[IP].src):
			first_time = False
			previous_timestamp = pkt.time
			
		elif (src_ip in pkt[IP].dst):
			now_timestamp = pkt.time
			timestamp_diff = now_timestamp - previous_timestamp
			
			time_interval_list.append(timestamp_diff)
			previous_timestamp = now_timestamp		
	
	if len(time_interval_list) == 0:
		min_biat = 0
		max_biat = 0
		mean_biat = 0
		std_biat = 0
	else:		
		min_biat = min(time_interval_list)
		max_biat = max(time_interval_list)
		mean_biat = np.mean(time_interval_list)
		std_biat = np.std(time_interval_list)
	
	return min_biat, max_biat, mean_biat, std_biat
	

# Flow Inter Arrival Time, the time between two packets sent in either direction (mean, min, max, std).
def extract_flowiat(pcap):
	time_interval_list = []
	first_time = True
	previous_timestamp = 0
	
	for pkt in pcap:
		if first_time:
			first_time = False
			previous_timestamp = pkt.time
			
		else:
			now_timestamp = pkt.time
			timestamp_diff = now_timestamp - previous_timestamp
			
			time_interval_list.append(timestamp_diff)
			previous_timestamp = now_timestamp		
	
	if len(time_interval_list) == 0:
		min_flowiat = 0
		max_flowiat = 0
		mean_flowiat = 0
		std_flowiat = 0
	else:
		min_flowiat = min(time_interval_list)
		max_flowiat = max(time_interval_list)
		mean_flowiat = np.mean(time_interval_list)
		std_flowiat = np.std(time_interval_list)	
	
	return min_flowiat, max_flowiat, mean_flowiat, std_flowiat
	

# The amount of time time a flow was active before going idle (mean, min, max, std). 
def extract_active(pcap):
	pass
	

# The amount of time time a flow was idle before becoming active (mean, min, max, std).
def extract_idle(pcap):
	pass


# Flow Bytes per second
def extract_fb_psec(pcap):
	total_bytes = 0
	for pkt in pcap:
		total_bytes += len(pkt)
	
	duration = pcap[-1].time - pcap[0].time
	
	if duration > 0:
		fb_psec = total_bytes / duration
	else:
		fb_psec = 0
		
	return fb_psec


# Flow packets per second
def extract_fp_psec(pcap):
	total_packet = len(pcap)
	duration = pcap[-1].time - pcap[0].time
	
	if duration > 0:
		fp_psec = total_packet / duration
	else:
		fp_psec = 0
	
	return fp_psec


# The duration of the flow
def extract_duration(pcap):
	duration = pcap[-1].time - pcap[0].time
	
	return duration
	
	
# Total forward packets number in one flow
def extract_fpn(pcap):
	forward_packet_num = 0
	
	src_ip = pcap[0][IP].src
	
	for pkt in pcap:
		if (src_ip in pkt[IP].src):
			forward_packet_num += 1	
	
	return forward_packet_num
	

# Total backward packets number in one flow
def extract_bpn(pcap):
	backward_packet_num = 0
	
	src_ip = pcap[0][IP].src
	
	for pkt in pcap:
		if(src_ip in pkt[IP].dst):
			backward_packet_num += 1
	
	return backward_packet_num
	

# Forward Flow Bytes per packet (mean/std/max/min)
def extract_Fwd_bytes_pkt(pcap):
	bytes_list = []
	
	src_ip = pcap[0][IP].src
	
	for pkt in pcap:
		if (src_ip in pkt[IP].src):
			bytes_list.append(len(pkt))
	
	if len(bytes_list) == 0:
		min_Fwd_bytes_pkt = 0
		max_Fwd_bytes_pkt = 0
		mean_Fwd_bytes_pkt = 0
		std_Fwd_bytes_pkt = 0
	else:
		min_Fwd_bytes_pkt  = min(bytes_list)
		max_Fwd_bytes_pkt  = max(bytes_list)
		mean_Fwd_bytes_pkt = np.mean(bytes_list)
		std_Fwd_bytes_pkt  = np.std(bytes_list)
	
	return min_Fwd_bytes_pkt, max_Fwd_bytes_pkt, mean_Fwd_bytes_pkt, std_Fwd_bytes_pkt
	

# Backward Flow Bytes per packet (mean/std/max/min)
def extract_Bwd_bytes_pkt(pcap):
	bytes_list = []
	
	src_ip = pcap[0][IP].src
	
	for pkt in pcap:
		if (src_ip in pkt[IP].dst):
			bytes_list.append(len(pkt))
	
	if len(bytes_list) == 0:
		min_Bwd_bytes_pkt = 0
		max_Bwd_bytes_pkt = 0
		mean_Bwd_bytes_pkt = 0
		std_Bwd_bytes_pkt = 0
	else:
		min_Bwd_bytes_pkt  = min(bytes_list)
		max_Bwd_bytes_pkt  = max(bytes_list)
		mean_Bwd_bytes_pkt = np.mean(bytes_list)
		std_Bwd_bytes_pkt  = np.std(bytes_list)

	return min_Bwd_bytes_pkt, max_Bwd_bytes_pkt, mean_Bwd_bytes_pkt, std_Bwd_bytes_pkt
	

# Count of Exchange Direction from Forward to Backward
def extract_Fw2Bw_EDC(pcap):
	Fw2Bw_count = 0
	
	src_ip = pcap[0][IP].src

	for i in range(len(pcap)-1):
		if (src_ip in pcap[i][IP].src) and (src_ip in pcap[i+1][IP].dst):
			Fw2Bw_count += 1
	
	return Fw2Bw_count
	

# Count of Exchange Direction from Backward to Forward
def extract_Bw2Fw_EDC(pcap):
	Bw2Fw_count = 0
	
	src_ip = pcap[0][IP].src

	for i in range(len(pcap)-1):
		if (src_ip in pcap[i][IP].dst) and (src_ip in pcap[i+1][IP].src):
			Bw2Fw_count += 1
	
	return Bw2Fw_count
	

# Every byte distribution from payload
def extract_byte_distribution(pcap):
	byte_distribution = [0 for i in range(256)]
	
	for pkt in pcap:
		if Raw in pkt:	
			payloads = str(pkt[Raw])	
			for payload in payloads:
				byte_distribution[ord(payload)] += 1

	return byte_distribution
	
	
# Forward byte/packet before change direction (mean/std/max/min)
def extract_Fwd_b_p_BCD(pcap):
	now_packet_num = 0
	now_bytes_num  = 0
	b_p_list = []
	src_ip = pcap[0][IP].src
	
	
	for pkt in pcap:
		if (src_ip in pkt[IP].src):
			now_packet_num += 1
			now_bytes_num  += len(pkt)
			
		elif (src_ip in pkt[IP].dst) and now_packet_num != 0 and now_bytes_num != 0:
			b_p_list.append(now_bytes_num/now_packet_num)
			now_packet_num = 0 
			now_bytes_num  = 0
	
	if len(b_p_list) == 0:
		min_Fwd_b_p_BCD = 0
		max_Fwd_b_p_BCD = 0
		mean_Fwd_b_p_BCD = 0
		std_Fwd_b_p_BCD = 0
	else:
		min_Fwd_b_p_BCD  = min(b_p_list)
		max_Fwd_b_p_BCD  = max(b_p_list)
		mean_Fwd_b_p_BCD = np.mean(b_p_list)
		std_Fwd_b_p_BCD  = np.std(b_p_list)	
	
	return min_Fwd_b_p_BCD, max_Fwd_b_p_BCD, mean_Fwd_b_p_BCD, std_Fwd_b_p_BCD
	

# Backward byte/packet before change direction (mean/std/max/min)
def extract_Bwd_b_p_BCD(pcap):
	now_packet_num = 0
	now_bytes_num  = 0
	b_p_list = []
	src_ip = pcap[0][IP].src
	
	
	for pkt in pcap:
		if (src_ip in pkt[IP].dst):
			now_packet_num += 1
			now_bytes_num  += len(pkt)
			
		elif (src_ip in pkt[IP].src) and now_packet_num != 0 and now_bytes_num != 0:
			b_p_list.append(now_bytes_num/now_packet_num)
			now_packet_num = 0 
			now_bytes_num  = 0
	
	if len(b_p_list) == 0:
		min_Bwd_b_p_BCD = 0
		max_Bwd_b_p_BCD = 0
		mean_Bwd_b_p_BCD = 0
		std_Bwd_b_p_BCD = 0
	else:
		min_Bwd_b_p_BCD  = min(b_p_list)
		max_Bwd_b_p_BCD  = max(b_p_list)
		mean_Bwd_b_p_BCD = np.mean(b_p_list)
		std_Bwd_b_p_BCD  = np.std(b_p_list)	
	
	return min_Bwd_b_p_BCD, max_Bwd_b_p_BCD, mean_Bwd_b_p_BCD, std_Bwd_b_p_BCD	
	
def extract_feature(full_filename):
	
	pcap = rdpcap(full_filename)
	
	try:
		chack_ip_layer = pcap[0][IP]
		chack_ip_layer = pcap[-1][IP]
	except:
		return None
		
	min_fiat, max_fiat, mean_fiat, std_fiat = extract_fiat(pcap) # 4 features
	min_biat, max_biat, mean_biat, std_biat = extract_biat(pcap) # 4 features
	min_flowiat, max_flowiat, mean_flowiat, std_flowiat = extract_flowiat(pcap) # 4 features
	fb_psec = extract_fb_psec(pcap) # 1 features
	fp_psec = extract_fp_psec(pcap) # 1 features
	min_Fwd_bytes_pkt, max_Fwd_bytes_pkt, mean_Fwd_bytes_pkt, std_Fwd_bytes_pkt = extract_Fwd_bytes_pkt(pcap) # 4 features
	min_Bwd_bytes_pkt, max_Bwd_bytes_pkt, mean_Bwd_bytes_pkt, std_Bwd_bytes_pkt = extract_Bwd_bytes_pkt(pcap) # 4 features
	Fw2Bw_count = extract_Fw2Bw_EDC(pcap) # 1 features
	Bw2Fw_count = extract_Bw2Fw_EDC(pcap) # 1 features	
	min_Fwd_b_p_BCD, max_Fwd_b_p_BCD, mean_Fwd_b_p_BCD, std_Fwd_b_p_BCD = extract_Fwd_b_p_BCD(pcap) # 4 features
	min_Bwd_b_p_BCD, max_Bwd_b_p_BCD, mean_Bwd_b_p_BCD, std_Bwd_b_p_BCD = extract_Bwd_b_p_BCD(pcap)	# 4 features
	
	all_feature = [min_fiat, max_fiat, mean_fiat, std_fiat, # 4 features
				   min_biat, max_biat, mean_biat, std_biat, # 4 features
				   min_flowiat, max_flowiat, mean_flowiat, std_flowiat, # 4 features
				   fb_psec, # 1 features
				   fp_psec, # 1 features
				   min_Fwd_bytes_pkt, max_Fwd_bytes_pkt, mean_Fwd_bytes_pkt, std_Fwd_bytes_pkt, # 4 features
				   min_Bwd_bytes_pkt, max_Bwd_bytes_pkt, mean_Bwd_bytes_pkt, std_Bwd_bytes_pkt, # 4 features
				   Fw2Bw_count, # 1 features
				   Bw2Fw_count, # 1 features
				   min_Fwd_b_p_BCD, max_Fwd_b_p_BCD, mean_Fwd_b_p_BCD, std_Fwd_b_p_BCD, # 4 features
				   min_Bwd_b_p_BCD, max_Bwd_b_p_BCD, mean_Bwd_b_p_BCD, std_Bwd_b_p_BCD] # 4 features
	
	return all_feature # 32 features
	
def main():
	
	if os.path.isdir(csv_dir) == False:
		os.mkdir(csv_dir)

	feature_names = ["min_fiat", "max_fiat", "mean_fiat", "std_fiat", # 4 features 
					 "min_biat", "max_biat", "mean_biat", "std_biat", # 4 features
					 "min_flowiat", "max_flowiat", "mean_flowiat", "std_flowiat", # 4 features
					 "fb_psec", # 1 features
					 "fp_psec", # 1 features
					 "min_Fwd_bytes_pkt", "max_Fwd_bytes_pkt", "mean_Fwd_bytes_pkt", "std_Fwd_bytes_pkt", # 4 features
					 "min_Bwd_bytes_pkt", "max_Bwd_bytes_pkt", "mean_Bwd_bytes_pkt", "std_Bwd_bytes_pkt", # 4 features
					 "Fw2Bw_count", # 1 features
					 "Bw2Fw_count", # 1 features
					 "min_Fwd_b_p_BCD", "max_Fwd_b_p_BCD", "mean_Fwd_b_p_BCD", "std_Fwd_b_p_BCD", # 4 features
					 "min_Bwd_b_p_BCD", "max_Bwd_b_p_BCD", "mean_Bwd_b_p_BCD", "std_Bwd_b_p_BCD", # 4 features
					 "class"]					  

	# generate zeroshot dataset
	zeroshot_data = []
	for dir_1 in zeroshot_class_dir:
		filenames = os.listdir(Pcap_dir+"zeroshot/"+dir_1)
		for filename in filenames:
			full_filename = Pcap_dir+"zeroshot/"+dir_1+filename			
			print(full_filename)
	
			row   = extract_feature(full_filename)
		
			if row == None:
				continue
			
			label = dir_1[:-1]
			row.append(label)
	
			zeroshot_data.append(row)
		
	zeroshot_df = pd.DataFrame(zeroshot_data, columns = feature_names)
	zeroshot_df.to_csv(csv_dir+zeroshot_handcraft_csv_filename, sep='\t')
	
if __name__ == '__main__':
    main()
