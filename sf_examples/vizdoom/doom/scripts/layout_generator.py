import math

WALL_SIZE = 1216.0

def hardcode(id:int, x:float, y:float):
        print(f"pos_x[{id}] = {x:.2f}; pos_y[{id}] = {y:.2f};")


# CIRCLE
def generate_layout2():
    tot_medkits = 24
    print(f"medkit_count = {tot_medkits};\n")
    sector_size = 360/tot_medkits
    sector_radian_angle = math.radians(sector_size)
    radius = 500
    middleX, middleY = WALL_SIZE/2.0, WALL_SIZE/2.0
    x, y = 1, 0
    for i in range(tot_medkits):
        x_aux = x*math.cos(sector_radian_angle) - y*math.sin(sector_radian_angle)
        y_aux = x*math.sin(sector_radian_angle) + y*math.cos(sector_radian_angle)
        x, y = x_aux, y_aux
        hardcode(i, middleX+x*radius, middleY+y*radius)

# SIN SHAPE
def generate_layout3():
    tot_medkits = 24
    print(f"medkit_count = {tot_medkits};\n")
    margin = 80
    shift_angle = math.radians(360/tot_medkits)
    peak_size = 350
    middleY = WALL_SIZE/2.0
    x, shift_x = margin, (WALL_SIZE-2*margin)/tot_medkits
    for i in range(tot_medkits):
        hardcode(i, x, middleY + math.sin(i*shift_angle)*peak_size)
        x += shift_x



# MATRIX
def generate_layout4():
    margin = 80
    grid_size = 5
    cell_size = (WALL_SIZE-2*margin)/5
    print(f"medkit_count = {grid_size*grid_size};\n")
    for i in range(grid_size):
        for j in range(grid_size):
            hardcode(i*grid_size+j, margin+i*cell_size+(cell_size/2), margin+j*cell_size+(cell_size/2))

if __name__ == "__main__":
    # generate_layout2()
    generate_layout3()
    # generate_layout4()