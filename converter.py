import math
import numpy as np
from PIL import Image

if __name__ == "__main__":
    flow = Image.open("flow.png", "r")
    flow = np.asarray(flow, dtype="int32")
    deposit = Image.open("deposit.png", "r")
    deposit = np.asarray(deposit, dtype="int32")
    wear = Image.open("wear.png", "r")
    wear = np.asarray(wear, dtype="int32")

    res = np.zeros((len(flow), len(flow[0]), 3))

    np.clip(res, 0, 255, out=res)
    data_u8 = res.astype('uint8')
    outimg = Image.fromarray(data_u8, "RGB")
    outimg.save("alpha.png")

    for x in range(0, len(flow)):
        print(x/len(flow)*100)
        for y in range(0, len(flow[0])):
            res[x, y, 0] = math.floor(math.sqrt(flow[x, y])/1.5)
            res[x, y, 1] = math.floor(math.sqrt(deposit[x, y]))
            res[x, y, 2] = math.floor(math.sqrt(wear[x, y]))

    np.clip(res, 0, 255, out=res)
    data_u8 = res.astype('uint8')
    outimg = Image.fromarray(data_u8, "RGB")
    outimg.save("res.png")
