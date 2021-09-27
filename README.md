# Malicious_flow_project

With increasing encrypted network flow in recent years, most internet traffic is protected using the cryptographic protocol known as Transport Layer Security (TLS). 
Some examples being HyperText Transfer Protocol Secure (HTTPS), File Transfer Protocol Secure Sockets Layer (FTPS), etc. 
Unfortunately, criminals with the intention of spreadingmalware have also adopted TLS to secure its data transmission. 
This trend makes threat detection more difficult because a common way to detect malware traffic, deep packet inspection (DPI), 
becomes ineffective if the traffic is encrypted.
Traditional packet analysis based network solutions for obvious reasons, cannot decipher the data inside encrypted traffic. Because of this shortcoming, 
it is difficult to ensure that encrypted traffic does not contain malicious data. 

Therefore, because decrypting is not an option, the use of passive detection can be used to collect feature differences in encryption suites and or other data traffic features between benign and malware data traffic.
Next, these collected features can be used in machine learning to train models to differentiate benign data traffic from malware data traffic. 
This method for solving the shortcomings of traditional packet analysis.
