import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def rel_perm(Sw, relperm):
    swr = relperm['water'][0]
    sor = relperm['oil'][0]
    Swd = (Sw - swr)/(1 - swr - sor)
    Sod = ((1-Sw) - sor)/(1 - swr - sor)

    if (type(Swd) == float):
        if Swd < 0: Swd = 0
        elif Swd > 1: Swd = 1

        else:
            Swd[Swd<0] = 0
            Swd[Swd>1] = 1

    nw = relperm['water'][2]
    no = relperm['oil'][2]
    krw = relperm['water'][1]*Swd**nw
    kro = relperm['oil'][1]*Sod**no
    return krw, kro


def front_saturation(Swf, relperm, fvisc):
    swr = relperm['water'][0]
    sor = relperm['oil'][0]
    wvisc = fvisc[0]
    ovisc = fvisc[1]
    # nw = relperm['water'][2]
    # no = relperm['oil'][2]
    dSwf = 1e-6
    krw1, kro1 = rel_perm(Swf, relperm)
    fw1 = krw1/wvisc/(krw1/wvisc + kro1/ovisc)
    krw1, kro1 = rel_perm(Swf+dSwf, relperm)
    fw2 = krw1/wvisc/(krw1/wvisc + kro1/ovisc)

    dfwf = (fw2-fw1)/dSwf
    g = dfwf - fw1/(Swf - swr)

    return g, dfwf

def return_right_type(Swf, relperm, fvisc):
    answer = front_saturation(Swf, relperm, fvisc)
    correct_return = answer[0][0], answer[1][0]
    # print(Swf," ")
    # print(correct_return)
    if Swf <0: return 10
    elif Swf>1: return 1000
    return correct_return


def bl(relperm, fvisc, tD, n_points):
    wvisc = fvisc[0]
    ovisc = fvisc[1]
    swr = relperm['water'][0]
    sor = relperm['oil'][0]
    # find front saturation
    Swf = fsolve(return_right_type, x0=0.4,
                 args=(relperm, fvisc))
    print(Swf)
    krwf, krof = rel_perm(Swf, relperm)

    fwf = krwf/wvisc/(krwf/wvisc + krof/ovisc)
    g, dfwf = front_saturation(Swf, relperm, fvisc)

    dsw1 = (1-sor - swr)/n_points
    Sw1 = np.r_[1-sor:swr:-dsw1]
    #  Sw1 = np.linspace(1-sor, swr, 102)
    NBL = len(Sw1)

    krw1, kro1 = rel_perm(Sw1, relperm)

    fw = krw1/wvisc/(krw1/wvisc + kro1/ovisc)


    xD = np.zeros(NBL)
    g, dfw = front_saturation(Sw1, relperm, fvisc)
    xD = tD*dfw

    for i in range(NBL):
        if (Sw1[i] < Swf):
            Sw1[i] =  swr
            xD[i] = tD*dfwf

    xD[0] = 0
    xD[-1] = 1
    return Sw1, xD



if __name__ == "__main__":
    wvisc = 0.38*1e-3
    ovisc = 1.03*1e-3
    tD = 0.177905
    n_points = 102
    corey_brooks = {
        # s_crit, kr_0, N
        'water': [0.2, 0.3, 2],
        'oil':   [0.4, 1.0, 2],
    }
    np.set_printoptions(edgeitems=1000,
                        linewidth=1000)
    result = bl(corey_brooks, [wvisc, ovisc], tD, n_points)
    Sw = result[0]
    xD = result[1]
    interp = interp1d(xD, Sw)
    xD = np.linspace(0, 1, n_points)
    Sw = interp(xD)
    #  print(result[0])
    #  print(xD)
    #  print(len(xD))
    #  with open ("analytical.txt", 'w') as f:
        #  for i in range(len(Sw)):
            #  f.write("%.4f"%Sw[i])
            #  if (i != len(Sw-1)):
                #  f.write("\n")

    plt.plot(xD, Sw, 'o')
    plt.show()
