#!/usr/bin/python
#coding:utf-8
import urllib2
import json
import time
import sys
import logging
import socket
import hashlib

from optparse import OptionParser
from OTXv2 import OTXv2
import IndicatorTypes
from scapy.all import *
import shutil
import dpkt

TLS_dir  = "/home/gavin/collect_splitter_pcap/"
malicious_TLS_dir = "result/"
#classes_dir  = ["EITest/", "Emotet/", "Hancitor/", "Nuclear/", "Rig/", "TrickBot/", "Dridex/", "Razy/", "HTBot/"]
#classes_dir = ["ctu17/"]
classes_dir = [sys.argv[1]+"/"]

url = 'http://ip.taobao.com/service/getIpInfo.php?ip='
API_KEY = '6e78a061abe61830b3e66c02850fa5ddfcdf8300df57184caf1fde70c65c8658'  #change your API_Key
OTX_SERVER = 'https://otx.alienvault.com/'

def get_pcap_ip(full_filename):
	pcap = rdpcap(full_filename)
	
        try:
            dst_ip = pcap[0][IP].dst
        except:
            dst_ip= pcap[0].dst
            #print("MAC Address")
	return dst_ip

def getValue(results, keys):
    if type(keys) is list and len(keys) > 0:

        if type(results) is dict:
            key = keys.pop(0)
            if key in results:
                return getValue(results[key], keys)
            else:
                return None
        else:
            if type(results) is list and len(results) > 0:
                return getValue(results[0], keys)
            else:
                return results
    else:
        return results

def CheckIp_malicious(otx, ip):
    alerts = []
    try:
        result = otx.get_indicator_details_by_section(IndicatorTypes.IPv4, ip, 'general')
    # if can't analyze ip address
    except:
    	return False    
    
    pulses = getValue(result, ['pulse_info', 'pulses'])    
    if pulses:
        for pulse in pulses:
            if 'name' in pulse:            	
                alerts.append('In pulse: ' + pulse['name'])
    
    if len(alerts) > 0:
        return True
    else:
        return False   

otx = OTXv2(API_KEY, server=OTX_SERVER)

if os.path.isdir(malicious_TLS_dir) == False:
	os.mkdir(malicious_TLS_dir)
		
count,j=0,0
a=[]

for class_dir in classes_dir:
        print(class_dir)
	if os.path.isdir(malicious_TLS_dir+class_dir) == False:
		os.mkdir(malicious_TLS_dir+class_dir)
		
	filenames = os.listdir(TLS_dir+class_dir)
	
	for i, filename in enumerate(filenames):
		#print(i)
		full_filename = TLS_dir + class_dir + filename
		ip = get_pcap_ip(full_filename)
                if CheckIp_malicious(otx, ip):
			output_dir = malicious_TLS_dir+class_dir
			shutil.copy2(full_filename, output_dir)

                        #print("filenames",full_filename)
                        reference = 'https://otx.alienvault.com/indicator/ip/'+ ip
                        #print ("potentially malicious file "+full_filename) # 
                        #print("reference:"+ reference)
                        if count==0:
                            a.append(ip)
                            count=count+1
                        
                        for num in range(len(a)):
                            if(ip!=a[num]):
                                j=j+1
                            if(j==len(a)):
                                a.append(ip)
                                count=count+1
                        j=0
                    
        print ("-----There are %d malicious hostname-----"%count)
        print(a)
