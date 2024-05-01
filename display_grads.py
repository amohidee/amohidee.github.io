from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np

def proc_file(img_path):
    x = 768
    y = 1024
    img = np.zeros((y,x))
    with open(img_path, 'r') as f:
        for i in range(y):
            for j in range(x):
                inp = int(float(f.readline()[:-1]))
                # inp = max(inp, 0)
                # inp = 500 if (inp > 300) else 0
                img[i][j] = inp
    plt.imshow(img)
    plt.show()
        

if __name__ == '__main__':
    fname = 'coins'
    imp = f"images/{fname}_gray.txt"
    proc_file(imp)
    imp = f"images/{fname}_grads.txt"
    proc_file(imp)
    imp = f"images/{fname}_nmsgrads.txt"
    proc_file(imp)
    imp = f"images/{fname}_radsym.txt"
    proc_file(imp)
    imp = f"images/{fname}_gauss.txt"
    proc_file(imp)
    