from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import sympy

from matplotlib.animation import FuncAnimation
from sympy.utilities.lambdify import implemented_function


steps = 200
t = np.linspace(0, 1, steps)

# Произвольные законы движения
phi = t * 2 * np.pi
psi = np.cos(t) * 3.3 * np.pi

figure = plt.figure(figsize=[9, 9])
subplot = figure.add_subplot(1, 1, 1)
subplot.axis('equal')
subplot.set_xlim((-3, 3))
subplot.set_ylim((-3, 3))

R_MAIN = 1
R_B = 0.1
L = 1.5

x_a = np.cos(phi)
y_a = np.sin(phi)

x_b = x_a + np.cos(psi) * L
y_b = y_a + np.sin(psi) * L

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
X_n = np.cos(radians) * radiuses
Y_n = np.sin(radians) * radiuses

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


animation = FuncAnimation(figure, update, frames=steps, interval=10, blit=True)

plt.show()
