# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 20:47:46 2020

@author: Evan Generoli
"""


# Day 1
#############################333
reset -sf
import pandas as pd
import math
import os
os.chdir(r'C:\Users\Evan Generoli\Documents')

df = pd.read_csv('AoCday1input.csv', header=None, names = ['mass'])

list(df.columns)
df.head()

df['mass'].apply(lambda x: math.floor(x/3) - 2).sum()


### pt 2
def fuel_req(mass):
    tot_fuel = 0
    x = math.floor(mass/3) - 2
    while x > 0:
        tot_fuel += x
        x = math.floor(x/3) - 2
    return tot_fuel
    
df['fuel'] = df['mass'].apply(fuel_req)
print(df['fuel'].sum())

# Day 2
#############################
reset -sf
import pandas as pd
import numpy as np

def run_intcode(code):
    i = 0
    while i < len(code):
        if code[i] == 1:
            code[code[i+3]] = code[code[i+1]] + code[code[i+2]]
        elif code[i] == 2:
            code[code[i+3]] = code[code[i+1]] * code[code[i+2]]
        else:
            break
        i += 4
    return code

testcode1 = np.array([1,0,0,0,99]) ##solution: 2,0,0,0,99 (1 + 1 = 2).
testcode2 = np.array([2,3,0,3,99]) ##solution: 2,3,0,6,99 (3 * 2 = 6).
testcode3 = np.array([2,4,4,5,99,0]) ##solution: 2,4,4,5,99,9801 (99 * 99 = 9801).
testcode4 = np.array([1,1,1,4,99,5,6,0,99]) ##solution: 30,1,1,4,2,5,6,0,99.

intcode_raw = [1,0,0,3,1,1,2,3,1,3,4,3,1,5,0,3,2,10,1,19,1,5,19,23,1,23,5,27,2,27,10,31,1,5,31,35,2,35,6,39,1,6,39,43,2,13,43,47,2,9,47,51,1,6,51,55,1,55,9,59,2,6,59,63,1,5,63,67,2,67,13,71,1,9,71,75,1,75,9,79,2,79,10,83,1,6,83,87,1,5,87,91,1,6,91,95,1,95,13,99,1,10,99,103,2,6,103,107,1,107,5,111,1,111,13,115,1,115,13,119,1,13,119,123,2,123,13,127,1,127,6,131,1,131,9,135,1,5,135,139,2,139,6,143,2,6,143,147,1,5,147,151,1,151,2,155,1,9,155,0,99,2,14,0,0]
intcode1 = np.array(intcode_raw)
intcode1[1] = 12
intcode1[2] = 2

out = run_intcode(intcode1)
out[0]

### pt 2

lst = [(i,j) for i in range(100) for j in range(100)]
output = []
for i in list(range(len(lst))):
    temp_code = np.array(intcode_raw)
    temp_code[1] = lst[i][0]
    temp_code[2] = lst[i][1]
    out = run_intcode(temp_code)
    output.append(out[0])

solution_index = output.index(19690720)
print(solution_index, lst[solution_index])


# Day 3
#############################
reset -sf
import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\Evan Generoli\Documents\AoCday3.csv')

wire1 = [(x[0],int(x[1:len(x)])) for x in df['wire1']]
wire2 = [(x[0],int(x[1:len(x)])) for x in df['wire2']]

corner_path1 = [(0,0)]
for i in range(len(wire1)):
    x_old = corner_path1[i][0]
    y_old = corner_path1[i][1]
    
    direction = wire1[i][0]
    delta = wire1[i][1]
    
    if direction == 'R':
        x_new = x_old + delta
        y_new = y_old      
    elif direction == 'L':
        x_new = x_old - delta
        y_new = y_old                       
    elif direction == 'U':
        x_new = x_old
        y_new = y_old + delta                
    elif direction == 'D':
        x_new = x_old
        y_new = y_old - delta               
    corner_path1.append((x_new, y_new))

def points_from_inst(path_instructions):
    path = [(0,0)]
    ind = 0
    for i in range(len(path_instructions)):
        x_old = path[ind][0]
        y_old = path[ind][1]
    
        direction = path_instructions[i][0]
        delta = path_instructions[i][1]
    
        for j in range(1,delta+1):
            if direction == 'R':
                x_new = x_old + j
                y_new = y_old      
            elif direction == 'L':
                x_new = x_old - j
                y_new = y_old                       
            elif direction == 'U':
                x_new = x_old
                y_new = y_old + j                
            elif direction == 'D':
                x_new = x_old
                y_new = y_old - j               
            path.append((x_new, y_new))
            ind += 1
    return path
        
path1 = points_from_inst(wire1)
path2 = points_from_inst(wire2)        
path1 = [[x[0],x[1]] for x in path1]
path2 = [[x[0],x[1]] for x in path2]

intersection_points = [x for x in path1 if x in path2]
intersection_points = list(filter(lambda x: x!= (0,0), intersection_points))
intersection_points_abs = [[abs(x[0]), abs(x[1])] for x in intersection_points]
intersection_points_dist = [sum(x) for x in intersection_points_abs]
min(intersection_points_dist)


### part 2
path1 = [tuple(x) for x in path1]
path2 = [tuple(x) for x in path2]
intersection_points = [tuple(x) for x in intersection_points]
steps1 = [path1.index(x) for x in intersection_points]
steps2 = [path2.index(x) for x in intersection_points]
#steps_tot = [x+y for x,y in zip(steps1, steps2)]
steps_tot = np.add(np.array(steps1), np.array(steps2))
min(steps_tot)


# Day 4
#################
reset -sf
import numpy as np
import pandas as pd

# 231832 - 767346

def non_decreasing(num):
    num = str(num)
    lst = []
    for i in range(len(num)):
        try:
            lst.append(int(num[i]) <= int(num[i+1]))
        except IndexError:
            continue
    return all(lst)
# or
def non_decreasing2(num):
    return all([int(str(num)[i]) <= int(str(num)[i+1]) for i in range(len(str(num))-1)])


def contains_double(num):
    num = str(num)
    doubles = [str(x)+str(x) for x in range(10)]
    lst = [doubles[i] in num for i in range(len(doubles))]
    return any(lst)
# or 
def contains_double2(num):
    num = str(num)
    lst = []
    for i in range(len(num)):
        try:
            lst.append(int(num[i]) == int(num[i+1]))
        except IndexError:
            continue
    return any(lst)

# input range is 231832 - 767346
x = range(231832, 767347)
out = list(filter(non_decreasing, x))
out = list(filter(contains_double, out))
len(out)

## part 2
def contains_double_only(num):
    num = str(num)
    doubles = [str(x)+str(x) for x in range(10)]
    triples = [str(x)+str(x)+str(x) for x in range(10)]
    lst = [doubles[i] in num and triples[i] not in num for i in range(len(doubles))]
    return any(lst)

out = list(filter(non_decreasing, x))
out = list(filter(contains_double_only, out))
len(out)


