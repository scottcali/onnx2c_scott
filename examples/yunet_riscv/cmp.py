import numpy as np,sys
a=np.fromfile("ref.bin",np.float32);b=np.fromfile(sys.argv[1],np.float32)
print("n",a.size,b.size,"maxabs",np.abs(a-b).max(),"maxrel",(np.abs(a-b)/(np.abs(a)+1e-3)).max())
