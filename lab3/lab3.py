from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import sympy

from matplotlib.animation import FuncAnimation
from sympy.utilities.lambdify import implemented_function
from scipy.integrate import odeint


ROT_OFFSET = -np.pi / 2

M1 = 3
M2 = 0.05
R = 1
L = 1
G = 9.8
C = 1
K = 0
PHI_M = 0


def func(y, t, m1, m2, r, l, g, c, k, phi_m):
    # y = [phi, psi, phi', psi']
    
    dy = np.zeros_like(y)
    dy[0] = y[2]
    dy[1] = y[3]

    # a11 * phi'' + a12 * psi'' = b1
    # a21 * phi'' + a22 * psi'' = b2
    a11 = ((m1 / 2) + m2) * r
    a12 = m2 * l * np.cos(y[0] - y[1])
    b1 = -m2 * l * y[3] ** 2 * np.sin(y[0] - y[1]) - m2 * g * np.sin(y[0]) - (c / r) * (y[0] - phi_m) + (k/r) * (dy[1] - dy[0])

    a21 = np.cos(y[0] - y[1])
    a22 = l / r
    b2 = y[2] ** 2 * np.sin(y[0] - y[1]) - (g / r) * np.sin(y[1]) - k * (y[3] - y[2]) / (m2 * r * l)

    dy[2] = (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a21)
    dy[3] = (a11 * b2 - a21 * b1) / (a11 * a22 - a12 * a21)

    return dy


PHI_0 = np.pi/2
PSI_0 = 0
D_PHI_0 = 0
D_PSI_0 = 0

steps = 1001
t = np.linspace(0, 10, steps)
y0 = np.array([PHI_0, PSI_0, D_PHI_0, D_PSI_0])

y = odeint(func, y0, t, args=(M1, M2, R, L, G, C, K, PHI_M))

phi = y[:,0]
psi = y[:,1]
d_phi = y[:,2]
d_psi = y[:,3]

dd_phi = np.zeros_like(t)
dd_psi = np.zeros_like(t)
r_ox = np.zeros_like(t)
r_oy = np.zeros_like(t)

for i in range(len(t)):
    local_dy = func(y[i], t[i], M1, M2, R, L, C, K, G, PHI_M)
    dd_phi[i] = local_dy[2]
    dd_psi[i] = local_dy[3]

    r_ox[i] = M2 * (R * (dd_phi[i] * np.cos(phi[i]) - (dd_phi[i] ** 2) * np.sin(phi[i])) + L * (dd_psi[i] * np.cos(phi[i]) - (d_psi[i] ** 2) * np.sin(psi[i])))
    r_oy[i] = M2 * (R * (dd_phi[i] * np.sin(phi[i]) - (dd_phi[i] ** 2) * np.cos(phi[i])) + L * (dd_psi[i] * np.sin(phi[i]) + (d_psi[i] ** 2) * np.cos(psi[i]))) + (M1 + M2) * G

fig = plt.figure(figsize=[17,9])
ax1 = fig.add_subplot(4, 2, 2)
ax1.plot(t, phi)
ax1.set_title("phi(t)")

ax2 = fig.add_subplot(4, 2, 4)
ax2.plot(t, psi)
ax2.set_title("psi(t)")

ax2 = fig.add_subplot(4, 2, 6)
ax2.plot(t, r_ox)
ax2.set_title("r_ox(t)")

ax2 = fig.add_subplot(4, 2, 8)
ax2.plot(t, r_oy)
ax2.set_title("r_oy(t)")

subplot = fig.add_subplot(1, 2, 1)
subplot.axis('equal')
subplot.set_xlim((-3, 3))
subplot.set_ylim((-3, 3))

R_MAIN = 1
R_B = 0.1
L = 1.5

x_a = np.cos(ROT_OFFSET + phi)
y_a = np.sin(ROT_OFFSET + phi)

x_b = x_a + np.cos(ROT_OFFSET + psi) * L
y_b = y_a + np.sin(ROT_OFFSET + psi) * L

# Центральная окружность
main_circle = Circle((0, 0), R_MAIN, fill=False)
subplot.add_patch(main_circle)

# Прямые AB и AO
line_ab = subplot.plot([x_a[0], x_b[0]], [y_a[0], y_b[0]], marker='o')[0]
line_oa = subplot.plot([0, x_a[0]], [0, y_a[0]], 'k--')[0]

# Точка B
b_circle = Circle((x_b[0], y_b[0]), R_B, fill=True, color='b', zorder=2)
b_circle_patch = subplot.add_patch(b_circle)

# Спираль
R_SPIRAL_1 = 0.05
R_SPIRAL_2 = 0.2

radians = np.linspace(0, 2 * np.pi, 50)
radiuses = np.linspace(R_SPIRAL_1, R_SPIRAL_2, 50)
X_n = np.cos(ROT_OFFSET + radians) * radiuses
Y_n = np.sin(ROT_OFFSET + radians) * radiuses

spiral = subplot.plot(X_n, Y_n)[0]


def Rot2D(X, Y, phi):
    # Поворот двумерной ДСК с помощью матрицы поворота
    X_r = X * np.cos(phi) - Y * np.sin(phi)
    Y_r = X * np.sin(phi) + Y * np.cos(phi)
    return X_r, Y_r


def update(i):
    print(i)

    line_ab.set_data([x_a[i], x_b[i]], [y_a[i], y_b[i]])
    line_oa.set_data([0, x_a[i]], [0, y_a[i]])
    b_circle.set_center((x_b[i], y_b[i]))
    
    new_X_n, new_Y_n = Rot2D(X_n, Y_n, phi[i])
    spiral.set_data(new_X_n, new_Y_n)

    return line_ab, line_oa, b_circle_patch, spiral


animation = FuncAnimation(fig, update, frames=steps, interval=10, blit=True)

plt.show()
