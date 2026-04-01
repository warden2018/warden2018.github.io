from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import math

#vector v is in the space and u is out of the space
def find_perpendicular_point(p_first,p_second,outer_point):
    v_x = p_second[0] - p_first[0]
    v_y = p_second[1] - p_first[1]
    u_x = outer_point[0] - p_first[0]
    u_y = outer_point[1] - p_first[1]

    if v_x==0 and v_y==0:
        return p_first
    else:
        per_point = [p_first[0],p_first[1]]
        inner_product = v_x * u_x + v_y * u_y
        lenSquare_v = v_x * v_x + v_y * v_y
        projected_v_x = inner_product / lenSquare_v * v_x
        projected_v_y = inner_product / lenSquare_v * v_y
        per_point[0] = projected_v_x + p_first[0]
        per_point[1] = projected_v_y + p_first[1]
        return per_point
    


fig = plt.figure(figsize=(4, 4))
ax = plt.axes()
fig.add_axes(ax)
#set x and y scale
ax.set_xlim(left=-2, right=3)
ax.set_ylim(bottom=-2, top=3)
#x and y axis without ticks and label
ax.arrow(0, -1, 0, 3, head_width=0.05, head_length=0.1, fc='k', ec='k')
ax.arrow(-1, 0, 2, 0, head_width=0.05, head_length=0.1, fc='k', ec='k')

#predicted point
predicted_point = {-0.5: 0.5}
predicted_point_x = list(predicted_point.keys())
predicted_point_y = list(predicted_point.values())
ax.scatter(predicted_point_x,predicted_point_y)
tex = r'$\hat x_{t| t-1}$'
ax.text(-1.5, 0.7, tex, fontsize=20, va='bottom')

#measurement line
measurement_space = [(-1, -1.3), (2, 2)]
(measurement_xs, measurement_ys) = zip(*measurement_space)
ax.add_line(Line2D(measurement_xs, measurement_ys, linewidth=2, color='red'))
tex = r'$\Omega = \left\{ x|Hx=z \right\}$'
ax.text(0.7, 0.1, tex, fontsize=15, va='bottom')
#predicted point perpendicular to measurement line
x_delta = find_perpendicular_point([measurement_space[0][0],measurement_space[0][1]],[measurement_space[1][0],
                                    measurement_space[1][1]],[predicted_point_x[0],predicted_point_y[0]])[0] - predicted_point_x[0]
y_delta = find_perpendicular_point([measurement_space[0][0],measurement_space[0][1]],[measurement_space[1][0],
                                    measurement_space[1][1]],[predicted_point_x[0],predicted_point_y[0]])[1] - predicted_point_y[0]                                 
ax.arrow(predicted_point_x[0],predicted_point_y[0],x_delta,y_delta, head_width=0.01, head_length=0.001, fc='k', ec='k')
tex = r'$\Delta x$'
ax.text(-0.2, 0.2, tex, fontsize=15, va='bottom')

plt.axis('off')
plt.show()