import math
from warnings import warn
class Sensitivity:
    SENSITIVITY = [
        [0,67.83744951,36.30248882,None],
        [1,14.69902315,8.565269959,None],
        [2,2.075609903,4.630562401,None],
        [3,1.201595399,4.080397002,None],
        [4,1.138682097,3.872669074,None],
        [5,1.051429274,3.745340131,None],
        [6,1.03525824,3.319938068,None],
        [7,1.063726464,3.25931204,None],
        [8,0.980352291,3.111523981,None],
        [9,0.885616402,2.972775966,None],
        [10,0.807906129,2.773642139,None],
        [11,0.685074112,2.030251882,None],
        [12,0.611636906,1.91325324,None],
        [13,0.578263135,1.501686683,None],
        [14,0.47444448,1.370216235,None],
        [15,0.408285516,0.872487485,None],
        [16,0.348999004,0.811455486,None],
        [17,0.302812523,0.712877571,None],
        [18,0.203259727,0.605241761,None],
        [19,0.07,0.21,0.092483912],   # 
        [20,None,None,0.066912479],
        [21,None,None,0.052314587],
        [22,None,None,0.039517244],
        [23,None,None,0.027816803],
        [24,None,None,0.020165959],
        [25,None,None,0.015381052],
        [26,None,None,0.013135206],
        [27,None,None,0.012424007],
        [28,None,None,0.011403062],
        [29,None,None,0.010871508],
        [30,None,None,0.0105951],
        [31,None,None,0.010218441],
        [32,None,None,0.009694533],
        [33,None,None,0.009518324],
        [34,None,None,0.009379172],
        [35,None,None,0.00961219],
        [36,None,None,0.009288778],
        [37,None,None,0.009528707],
        [38,None,None,0.009085895],
        [39,None,None,0.009030208],
        [40,None,None,0.008821486],
        [41,None,None,0.00882366],
        [42,None,None,0.00870809],
        [43,None,None,0.008548812],
        [44,None,None,0.00826456],
        [45,None,None,0.007971426],
        [46,None,None,0.008040328],
        [47,None,None,0.007697782],
        [48,None,None,0.007308777],
        [49,None,None,0.007087144],
        [50,None,None,0.007204945],
        [51,None,None,0.00703193],
        [52,None,None,0.006515885],
        [53,None,None,0.006174168],
        [54,None,None,0.005732876],
        [55,None,None,0.004940999],
        [56,None,None,0.004144642],
        [57,None,None,0.004232011]
    ]

    IMG = 1
    TXT = 2
    X = 3

    @classmethod
    def impact_of_mseloss(cls, before_layer:int, element:int, mse_loss:float):
        sensitivity = None
        if element==1 or element==2:
            if before_layer==19:
                warn("img and txt before layer 19 (after 18) are approximated from x")
            if before_layer>=0 and before_layer<=19:
                sensitivity = cls.SENSITIVITY[before_layer][element]
            else:
                raise Exception(f"img or txt don't exist before layer {before_layer}")
        if element==3:
            if before_layer<19:
                warn("x before a double layer is approximated from img and txt")
                sensitivity = cls.SENSITIVITY[before_layer][0] * 0.8 + cls.SENSITIVITY[before_layer][1] * 0.2
            else:
                sensitivity = cls.SENSITIVITY[before_layer][element]
        if sensitivity is None:
            raise Exception(f"Couldn't get a sensitivity for {element} {before_layer}")

        return math.sqrt(mse_loss) * sensitivity
        