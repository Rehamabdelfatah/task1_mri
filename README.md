# Team members
## Reham Abdelfatah
## Renad Taher
## Nancy Salah 
## Gehad Mohammed


# Rotation  of the bulk magnetization vector
## Code:
import numpy as np<br />
import matplotlib.pyplot as plt<br />
import math<br />
import time<br />
import pylab<br />
dT = 1	<br />
T = 1000<br />
df = 10<br />
T1 = 600<br />
T2 = 100<br />
N = math.ceil(T/dT)+1<br />
result=[None]*2<br />
def freepression(T,T1,T2,df):<br />
  phi = 2*math.pi*df*T/1000<br />
  Rz = [[math.cos(phi), -math.sin(phi), 0],
      [math.sin(phi), math.cos(phi) ,0],
      [0, 0, 1]]<br />
  E1 = math.exp(-T/T1)	<br />
  E2 = math.exp(-T/T2)<br />
  B = [0, 0, 1-E1]<br />
  A = [[E2, 0, 0],
       [0 ,E2, 0],
       [0, 0 ,E1]]<br />
  resultA = np.dot(A,Rz)<br />
  return (resultA,B	)<br />
def decay():<br />
  A,B = freepression(dT,T1,T2,df)<br />
  M = np.zeros((N,3))<br />
  M[0,:]= np.array([1,0,0])<br />
  for i in range (1,N):<br />
    M[i,:] = np.dot(A,M[i-1,:]) + B<br />
  return (M)<br />
M = decay()<br />
pylab.subplot(111)<br />
timedata = np.arange(N)<br />
axes = pylab.gca()<br />
axes.set_xlim(0,1000)<br />
axes.set_xlabel("time")<br />
axes.set_ylim(-1,1)<br />
axes.set_ylabel("Mx,My,Mz")<br />
Mx = M[:,0]<br />
My = M[:,1]<br />
Mz = M[:,2]<br />
plt.plot(timedata,Mx)<br />
plt.plot(timedata,My)<br />
plt.plot(timedata,Mz)<br />
plt.show()<br />
## Results :
![alt text](MXYZ.png)
# Bulk magnetization’s trajectory
### The vector is in the Mxy plane and start decaying and increases in Mz direction 
## Code:
import numpy as np<br />
import matplotlib.pyplot as plt<br />
import math<br />
import time<br />
import pylab<br />
dT = 1	<br />
T = 1000<br />
df = 10<br />
T1 = 600<br />
T2 = 100<br />
N = math.ceil(T/dT)+1<br />
result=[None]*2<br />
def freepression(T,T1,T2,df):<br />
  phi = 2*math.pi*df*T/1000<br />
  Rz = [[math.cos(phi), -math.sin(phi), 0],
      [math.sin(phi), math.cos(phi) ,0],
      [0, 0, 1]]<br />
  E1 = math.exp(-T/T1)	<br />
  E2 = math.exp(-T/T2)<br />
  B = [0, 0, 1-E1]<br />
  A = [[E2, 0, 0],
       [0 ,E2, 0],
       [0, 0 ,E1]] <br />
  resultA = np.dot(A,Rz)<br />
  return (resultA,B	)<br />
#resultA,B = freepression(T,T1,T2,df)<br />
def decay():<br />
  A,B = freepression(dT,T1,T2,df)<br />
  M = np.zeros((N,3))<br />
  M[0,:]= np.array([1,0,0])<br />
  for i in range (1,N): <br />
    M[i,:] = np.dot(A,M[i-1,:]) + B <br />
  return (M)<br />
M = decay()<br />
pylab.subplot(111)<br />
xdata = []<br />
ydata = []<br />
timedata = np.arange(N)<br />
axes = pylab.gca()<br />
axes.set_xlim(-10,10)<br />
axes.set_ylim(-10,10)<br />
line,=axes.plot(xdata,ydata,'r-')<br />
for i in range(N) :<br />
  xdata.append(M[i,0])<br />
  ydata.append(M[i,1])<br />
  line.set_xdata(xdata)<br />
  line.set_ydata(ydata)<br />
  plt.draw()<br />
  plt.pause(1e-17)<br />
  time.sleep(0.01)<br />
plt.show() <br />

## Results :

![alt text](image0.gif)

# Fourier transform of an image
## Code:

import numpy as np <br />
from PIL import Image <br />
image = Image.open('mri.jpeg')<br />
#image.show()<br /> <!--- if you want to display the real image remove the # -->
imgByte = np.asarray(image)<br />
ft = np.fft.fft2(imgByte)<br />
Ift = ft.astype(np.uint8)<br />
imageft = Image.fromarray(Ift).save('ft.jpg')<br />
fourir = Image.open('ft.jpg')<br />
fourir.show()<br />

## Results :
### The real image:
![alt text](mri.jpeg)
### The fourier result:
![alt text](ft.jpg)

# The non-uniformity effect
### This graph shows random distrbution of bo along human body in z direction using mri of 1.5 Tesla
## code :
from random import shuffle<br />
import matplotlib.pyplot as plt<br />
RandomList = [[i] for i in range(1000,1500)]<br /> 
shuffle (RandomList)<br />
plt.plot(RandomList)<br />
plt.show()<br />
## Results:
![alt text](Bzplot.jpg) <!--- where the scale is mT --> 

