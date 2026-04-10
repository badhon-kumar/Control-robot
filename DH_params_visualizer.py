import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def dh_matrix(theta_rad, d, a, alpha_rad):
    """Calculates the standard Denavit-Hartenberg transformation matrix."""
    ct = np.cos(theta_rad)
    st = np.sin(theta_rad)
    ca = np.cos(alpha_rad)
    sa = np.sin(alpha_rad)
    
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,  sa,     ca,    d   ],
        [ 0,  0,      0,     1   ]
    ])

def draw_coordinate_frame(ax, T, name="", length=40, line_width=3):
    """Draws X(Red), Y(Green), and Z(Blue) axes based on a 4x4 matrix."""
    # The origin is the 4th column of the matrix
    origin = T @ np.array([0, 0, 0, 1])
    
    # Calculate the tips of the X, Y, and Z arrows
    x_tip = T @ np.array([length, 0, 0, 1])
    y_tip = T @ np.array([0, length, 0, 1])
    z_tip = T @ np.array([0, 0, length, 1])
    
    # Plot the axes (Red=X, Green=Y, Blue=Z)
    ax.plot([origin[0], x_tip[0]], [origin[1], x_tip[1]], [origin[2], x_tip[2]], 'r', linewidth=line_width)
    ax.plot([origin[0], y_tip[0]], [origin[1], y_tip[1]], [origin[2], y_tip[2]], 'g', linewidth=line_width)
    ax.plot([origin[0], z_tip[0]], [origin[1], z_tip[1]], [origin[2], z_tip[2]], 'b', linewidth=line_width)
    
    if name:
        ax.text(origin[0], origin[1], origin[2] - 15, name, color='black', fontweight='bold')
        
    return origin

# ─── SETUP MATPLOTLIB FIGURE ─────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.35) # Make room for sliders at the bottom
fig.canvas.manager.set_window_title("Denavit-Hartenberg Explorer")

# ─── SETUP SLIDERS ───────────────────────────────────────────────────────
ax_theta1 = plt.axes([0.15, 0.25, 0.3, 0.03])
ax_d1     = plt.axes([0.15, 0.20, 0.3, 0.03])
ax_a1     = plt.axes([0.15, 0.15, 0.3, 0.03])
ax_alpha1 = plt.axes([0.15, 0.10, 0.3, 0.03])

ax_theta2 = plt.axes([0.6, 0.25, 0.3, 0.03])
ax_d2     = plt.axes([0.6, 0.20, 0.3, 0.03])
ax_a2     = plt.axes([0.6, 0.15, 0.3, 0.03])
ax_alpha2 = plt.axes([0.6, 0.10, 0.3, 0.03])

# (axes, label, min_val, max_val, initial_val)
slider_theta1 = Slider(ax_theta1, 'Theta 1', -180, 180, valinit=45)
slider_d1     = Slider(ax_d1,     'd1',    0, 200, valinit=100)
slider_a1     = Slider(ax_a1,     'a1',    0, 200, valinit=100)
slider_alpha1 = Slider(ax_alpha1, 'Alpha 1', -180, 180, valinit=0)

slider_theta2 = Slider(ax_theta2, 'Theta 2', -180, 180, valinit=45)
slider_d2     = Slider(ax_d2,     'd2',    0, 200, valinit=100)
slider_a2     = Slider(ax_a2,     'a2',    0, 200, valinit=100)
slider_alpha2 = Slider(ax_alpha2, 'Alpha 2', -180, 180, valinit=0)

def update(val):
    """Called every time a slider is moved."""
    ax.cla() # Clear the 3D plot
    
    # Lock the camera view boundaries so it doesn't jump around
    ax.set_xlim([-400, 400])
    ax.set_ylim([-400, 400])
    ax.set_zlim([0, 500])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 1. Get slider values
    theta1_rad = np.radians(slider_theta1.val)
    d1         = slider_d1.val
    a1         = slider_a1.val
    alpha1_rad = np.radians(slider_alpha1.val)
    
    theta2_rad = np.radians(slider_theta2.val)
    d2         = slider_d2.val
    a2         = slider_a2.val
    alpha2_rad = np.radians(slider_alpha2.val)
    
    # 2. Calculate the Matrices
    T_base = np.eye(4) # Identity matrix (0,0,0)
    T_1 = dh_matrix(theta1_rad, d1, a1, alpha1_rad)
    T_2 = T_1 @ dh_matrix(theta2_rad, d2, a2, alpha2_rad)
    
    # 3. Draw the Coordinate Frames
    origin_base = draw_coordinate_frame(ax, T_base, "Base Motor (0)", length=50, line_width=2)
    origin_1 = draw_coordinate_frame(ax, T_1, "Motor (1)", length=50, line_width=3)
    origin_2 = draw_coordinate_frame(ax, T_2, "Motor (2)", length=60, line_width=4)
    
    # 4. Draw the physical "Links" connecting them
    ax.plot([origin_base[0], origin_1[0]], 
            [origin_base[1], origin_1[1]], 
            [origin_base[2], origin_1[2]], 
            color='black', linestyle='--', linewidth=2, label="Link 1")
            
    ax.plot([origin_1[0], origin_2[0]], 
            [origin_1[1], origin_2[1]], 
            [origin_1[2], origin_2[2]], 
            color='purple', linestyle='--', linewidth=2, label="Link 2")
    
    ax.legend(loc="upper left")
    fig.canvas.draw_idle()

# Attach the update function to the sliders
slider_theta1.on_changed(update)
slider_d1.on_changed(update)
slider_a1.on_changed(update)
slider_alpha1.on_changed(update)

slider_theta2.on_changed(update)
slider_d2.on_changed(update)
slider_a2.on_changed(update)
slider_alpha2.on_changed(update)

# Draw the initial state
update(0)

# Display the window
plt.show()