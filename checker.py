import socket
from classer import *
import re

threshold = input()
content_buffer = [] 
s = socket.socket()         
s.bind(('0.0.0.0', 8090 ))
s.listen(0)                 
 
while True:
 
    client, addr = s.accept()
 
    while True:
        content = client.recv(32)
 
        if len(content) ==0:
           break
        elif len(content_buffer) > threshold :
            break;
        else:
            digit = re.findall(r'\d+',content.decode("utf-8"))
            content_buffer.append(digit)
            print(content)

    print(content_buffer) 
    print("Closing connection")
    appliance,time_used = lister(content_buffer,17)
    print(appliance,time_used)
    client.close()
