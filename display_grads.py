from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np

# x  = 768
# y = 1024
fname = 'coins'

with open(f"images/{fname}.txt",'r') as f:
    ln = (f.readline()[:-1]).split(" ")
    x = int(ln[0])
    y = int(ln[1])
    print(y, x)

def proc_file(img_path):
    print(img_path)
    img = np.zeros((y,x))
    with open(img_path, 'r') as f:
        for i in range(y):
            for j in range(x):
                inp = int(float(f.readline()[:-1]))
                # inp = max(inp, 0)
                # inp = 500 if (inp > 300) else 0
                if("radsym" in img_path):
                    # inp = 0 if (inp < 1000) else inp
                    inp = inp
                img[i][j] = inp + 1
    plt.imshow(img)
    plt.show()

def proc_side():
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    xg = np.zeros((y,x))
    yg = np.zeros((y,x))
    dir = np.zeros((y,x))
    nms = np.zeros((y,x))
    img_path = f"images/{fname}_nmsgrads.txt"
    with open(img_path, 'r') as f:
        for i in range(y):
            for j in range(x):
                inp = int(float(f.readline()[:-1]))
                nms[i][j] = inp

    img_path = f"images/{fname}_xgr.txt"
    with open(img_path, 'r') as f:
        for i in range(y):
            for j in range(x):
                inp = int(float(f.readline()[:-1]))
                xg[i][j] = inp if(nms[i][j] != 0) else 0
    img_path = f"images/{fname}_ygr.txt"
    with open(img_path, 'r') as f:
        for i in range(y):
            for j in range(x):
                inp = int(float(f.readline()[:-1]))
                yg[i][j] = inp if(nms[i][j] != 0) else 0
    for i in range(y):
        for j in range(x):
            dir[i][j] = np.arctan2(yg[i][j],xg[i][j]) * 180 /np.pi
    ax1.imshow(yg)
    ax2.imshow(xg)
    ax3.imshow(dir)
    plt.show()

        

if __name__ == '__main__':
    print("gray")
    imp = f"images/{fname}_gray.txt"
    proc_file(imp)
    print("grads")
    imp = f"images/{fname}_grads.txt"
    proc_file(imp)
    print("nms")
    imp = f"images/{fname}_nmsgrads.txt"
    proc_file(imp)
    print("radsym")
    imp = f"images/{fname}_radsym.txt"
    proc_file(imp)
    print("gauss")
    imp = f"images/{fname}_gauss.txt"
    proc_file(imp)
    print("gaussnms")
    imp = f"images/{fname}_gaussnms.txt"
    proc_file(imp)
    print("dir grads")
    proc_side()
    