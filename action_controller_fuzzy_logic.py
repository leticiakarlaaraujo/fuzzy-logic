import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

# 1. Define problem variables and their universes of discourse
speed = ctrl.Antecedent(np.arange(0, 1001, 1), 'speed')
throttle_input = ctrl.Antecedent(np.arange(0, 101, 1), 'throttle_input')
# Universe of discourse for actions (-1: Brake, 0: Neutral, 1: Accelerate)
action = ctrl.Consequent(np.arange(-1, 2, 1), 'action')

# 2. Define membership functions for each variable
speed['low'] = fuzz.trimf(speed.universe, [0, 0, 500])
speed['medium'] = fuzz.trimf(speed.universe, [0, 500, 1001])
speed['high'] = fuzz.trimf(speed.universe, [500, 1001, 1001])

throttle_input['low'] = fuzz.trimf(throttle_input.universe, [0, 0, 30])
throttle_input['medium'] = fuzz.trimf(throttle_input.universe, [20, 50, 80])
throttle_input['high'] = fuzz.trimf(throttle_input.universe, [70, 101, 101])

action['brake'] = fuzz.trimf(action.universe, [-1, -1, 0])
action['neutral'] = fuzz.trimf(action.universe, [-1, 0, 1])
action['accelerate'] = fuzz.trimf(action.universe, [0, 1, 1])

# 3. Define fuzzy rules
rule1 = ctrl.Rule(speed['low'] & throttle_input['low'], action['accelerate'])
rule2 = ctrl.Rule(speed['low'] & throttle_input['medium'], action['accelerate'])
rule3 = ctrl.Rule(speed['low'] & throttle_input['high'], action['neutral'])
rule4 = ctrl.Rule(speed['medium'] & throttle_input['low'], action['accelerate'])
rule5 = ctrl.Rule(speed['medium'] & throttle_input['medium'], action['neutral'])
rule6 = ctrl.Rule(speed['medium'] & throttle_input['high'], action['brake'])
rule7 = ctrl.Rule(speed['high'] & throttle_input['low'], action['neutral'])
rule8 = ctrl.Rule(speed['high'] & throttle_input['medium'], action['brake'])
rule9 = ctrl.Rule(speed['high'] & throttle_input['high'], action['brake'])

# 4. Create the fuzzy control system
action_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
action_simulation = ctrl.ControlSystemSimulation(action_control)

# 5. Provide inputs to the system
input_speed_val = 300
input_throttle_val = 70

action_simulation.input['speed'] = input_speed_val
action_simulation.input['throttle_input'] = input_throttle_val

# 6. Compute the output
action_simulation.compute()
print("Action Output: ", action_simulation.output['action'])
print("")

# 7. Visualize fuzzy sets and the output
speed.view()
plt.axvline(x=input_speed_val, color='r', linestyle='--') # Mark current speed input
plt.title("Speed Membership Functions") # Added title for clarity
print("")

throttle_input.view()
plt.axvline(x=input_throttle_val, color='r', linestyle='--') # Mark current throttle input
plt.title("Throttle Input Membership Functions") # Added title for clarity

action.view(sim=action_simulation)
plt.title("Action Output and Membership") # Added title for clarity
#plt.axvline(x=action_simulation.output['action'], color='g', linestyle='--') # Marks the defuzzified action output

plt.show()