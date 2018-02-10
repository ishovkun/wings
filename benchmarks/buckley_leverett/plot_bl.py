# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import sys
import os

from analytical import bl
from retreive_paraview_data import retreive_data

corey_brooks = {
    # s_crit, kr_0, N
    'water': [0.2, 0.3, 2],
    'oil':   [0.4, 1.0, 2],
}
wvisc = 0.38*1e-3
ovisc = 1.03*1e-3
tD = 0.177905
n_points = 102
np.set_printoptions(edgeitems=1000,
                    linewidth=1000)
result = bl(corey_brooks, [wvisc, ovisc], tD, n_points)
Sw = result[0]
xD = result[1]
interp = interp1d(xD, Sw)
xD = np.linspace(0, 1, n_points)
Sw = interp(xD)

# retreive data from pvd file
pvd_file = sys.argv[1]
csv_file = "numerical_solution.csv"
retreive_data(pvd_file, csv_file, n_points)
data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
os.remove(csv_file)

x_num = data[:, 3]
p_num = data[:, 0]
Sw_num = data[:, 1]

xD_num = x_num/max(x_num)
# ft = 0.3048
# q_rate = 5.61*50.
# area = 25.*50.*ft^2
# length = 510*ft
# phi = 0.25
# tD_num =

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

fig = Figure(figsize=(8,8))
ax = fig.add_subplot(111)

ax.plot(xD, Sw, 'k')
ax.plot(xD_num, Sw_num, 'ko-')

# plt.plot(xD, Sw, 'k')
# plt.plot(xD_num, Sw_num, 'bo-')

ax.set_xlabel("Dimensionless coordinate")
ax.set_ylabel("Water saturation")


if not os.path.exists('benchmark_results'):
   os.mkdir('benchmark_results')

canvas = FigureCanvasAgg(fig)
canvas.print_figure("benchmark_results/buckley-leverett.png", dpi=96)

# plt.savefig('benchmark_results/buckley-leverett.pdf')
# plt.show()
