import math
import numpy
import skfuzzy.control
import matplotlib.pyplot as plt

def Car_Kinematics_Model(x, y, phi, theta):
    phi = phi / 180 * math.pi
    theta = theta / 180 * math.pi
    x = x + math.cos(phi + theta) + math.sin(phi) * math.sin(theta)
    y = y + math.sin(phi + theta) - math.sin(phi) * math.sin(theta)
    phi = phi - math.asin(2 * math.sin(theta) / 4)  # 4為車長
    phi = phi / math.pi * 180
    return x, y, phi

def Fuzzy_Controller(V1, V2, V3):  # V1: delta_x, V2: delta_phi, V3: delta_y

    universe_delta_x=numpy.arange(-11,11,0.1) # delta x
    universe_delta_phi=numpy.arange(-101,101,0.1) # delta phi
    universe_theta=numpy.arange(-16,16.1,0.1) # theta
    universe_delta_y=numpy.arange(-200, 200.1, 1)

    delta_y = skfuzzy.control.Antecedent(universe_delta_y, 'delta_y')
    delta_x=skfuzzy.control.Antecedent(universe_delta_x,'delta_x')
    delta_phi=skfuzzy.control.Antecedent(universe_delta_phi,'delta_phi')
    theta=skfuzzy.control.Consequent(universe_theta,'theta')

    delta_x['A11'] = skfuzzy.trapmf(universe_delta_x, [-11, -11, -5, -4])  # 極左
    delta_x['A12'] = skfuzzy.trimf(universe_delta_x, [-8, -1, 0])          # 左
    delta_x['A13'] = skfuzzy.trimf(universe_delta_x, [-6, 0, 6])           # 左右偏中
    delta_x['A14'] = skfuzzy.trimf(universe_delta_x, [0, 1, 8])            # 右
    delta_x['A15'] = skfuzzy.trapmf(universe_delta_x, [4, 5, 11, 11])      # 極右

    delta_phi['A21'] = skfuzzy.trapmf(universe_delta_phi, [-90, -90, -17, -15])       # 車尾極左
    delta_phi['A22'] = skfuzzy.trimf(universe_delta_phi, [-36, -3, 0])               # 車尾大左
    delta_phi['A23'] = skfuzzy.trimf(universe_delta_phi, [-35, 0, 35])                # 車尾直行
    delta_phi['A24'] = skfuzzy.trimf(universe_delta_phi, [0, 3, 36])                 # 車尾大右
    delta_phi['A25'] = skfuzzy.trapmf(universe_delta_phi, [15, 17, 90, 90])           # 車尾極右

    delta_y['A31'] = skfuzzy.trapmf(universe_delta_y, [187, 188, 200, 201])
    delta_y['A32'] = skfuzzy.trimf(universe_delta_y, [0, 30, 100])
    delta_y['A33'] = skfuzzy.trimf(universe_delta_y, [20, 25, 185])
    delta_y['A34'] = skfuzzy.trapmf(universe_delta_y, [183, 184, 187, 188])    

    theta['A41'] = skfuzzy.trimf(universe_theta, [-16, -15, -6])   # 大左
    theta['A42'] = skfuzzy.trimf(universe_theta, [-11, -11, 0])     # 小左
    theta['A43'] = skfuzzy.trimf(universe_theta, [-6, 0, 6])       # 直行
    theta['A44'] = skfuzzy.trimf(universe_theta, [0, 11, 11])       # 小右
    theta['A45'] = skfuzzy.trimf(universe_theta, [6, 15, 16])      # 大右

    theta.defuzzify_method='centroid' # 重心法

    delta_x.view()
    delta_phi.view()
    delta_y.view()
    theta.view()
    plt.show()

    # rule base
    rule1=skfuzzy.control.Rule(antecedent=((delta_x['A11']&delta_phi['A23']&delta_y['A33'])|
                                           (delta_x['A12']&delta_phi['A23']&delta_y['A33'])|
                                           (delta_x['A12']&delta_phi['A24']&delta_y['A33'])|
                                           (delta_x['A13']&delta_phi['A24']&delta_y['A33'])|
                                           (delta_x['A12']&delta_phi['A24']&delta_y['A32'])),consequent=theta['A44'],label='turning right')
    rule2=skfuzzy.control.Rule(antecedent=((delta_x['A11']&delta_phi['A22']&delta_y['A33'])|
                                           (delta_x['A13']&delta_phi['A23']&delta_y['A33'])|
                                           (delta_x['A14']&delta_phi['A24']&delta_y['A33'])|
                                           (delta_x['A13']&delta_phi['A23']&delta_y['A32'])),consequent=theta['A43'],label='turning straight')
    rule3=skfuzzy.control.Rule(antecedent=((delta_x['A13']&delta_phi['A22']&delta_y['A33'])|
                                           (delta_x['A12']&delta_phi['A22']&delta_y['A33'])|
                                           (delta_x['A14']&delta_phi['A22']&delta_y['A33'])|
                                           (delta_x['A14']&delta_phi['A23']&delta_y['A33'])|
                                           (delta_x['A14']&delta_phi['A22']&delta_y['A32'])),consequent=theta['A42'],label='turning left')
    rule4=skfuzzy.control.Rule(antecedent=((delta_x['A11']&delta_phi['A23']&delta_y['A31'])),consequent=theta['A45'],label='turning hard right')
    rule5=skfuzzy.control.Rule(antecedent=((delta_x['A11']&delta_phi['A23']&delta_y['A34'])|
                                           (delta_x['A12']&delta_phi['A23']&delta_y['A34'])|
                                           (delta_x['A11']&delta_phi['A22']&delta_y['A34'])|
                                           (delta_x['A12']&delta_phi['A22']&delta_y['A34'])),consequent=theta['A41'],label='turning hard left')


    system = skfuzzy.control.ControlSystem(rules=[rule1, rule2, rule3, rule4, rule5])
    sim = skfuzzy.control.ControlSystemSimulation(system)

    sim.input['delta_x'] = V1
    sim.input['delta_phi'] = V2
    sim.input['delta_y'] = V3

    try:
        sim.compute()
        return sim.output['theta']
    except Exception as e:
        print(f"Error in fuzzy computation: {e}")
        return 0  


# Initial values
x = -10
y = 600
phi = 90

x_pos = [0] * 200
y_pos = [0] * 200

for t in range(0,200,1):
    delta_x = x - 0
    delta_phi = phi - 90
    delta_y = abs(y - 800)
    theta=Fuzzy_Controller(delta_x, delta_phi, delta_y)
    x,y,phi=Car_Kinematics_Model(x,y,phi,theta)
    x_pos[t]=x
    y_pos[t]=y

plt.plot(x_pos,y_pos)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Car path (with fuzzy control)")
plt.grid(True)
plt.show()