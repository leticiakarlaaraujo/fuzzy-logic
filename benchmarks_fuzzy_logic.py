import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

input_profitability = int(input("Define de profitability: "))
input_risk = int(input("Define de risk: "))

profitability = ctrl.Antecedent(np.arange(-10, 20.1, 0.1), 'profitability')  
risk = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'risk')        

decision = ctrl.Consequent(np.arange(0, 10.1, 0.1), 'decision') 

profitability['Low'] = fuzz.trimf(profitability.universe, [-10, -5, 0])
profitability['Medium'] = fuzz.trimf(profitability.universe, [0, 5, 10])
profitability['High'] = fuzz.trimf(profitability.universe, [10, 15, 20])

risk['Low'] = fuzz.trimf(risk.universe, [0, 1, 3])
risk['Medium'] = fuzz.trimf(risk.universe, [2, 5, 7])
risk['High'] = fuzz.trimf(risk.universe, [6, 8, 10])

decision['Bad'] = fuzz.trimf(decision.universe, [0, 2, 4])
decision['Neutral'] = fuzz.trimf(decision.universe, [3, 5, 7])
decision['Good'] = fuzz.trimf(decision.universe, [6, 8, 10])

profitability.view()
plt.title('Profitability Relevance Function') 

risk.view()
plt.title('Risk Relevance Functions')

decision.view()
plt.title('Decision Relevance Functions') 

rule1 = ctrl.Rule(profitability['High'] & risk['Low'], decision['Good'])
rule2 = ctrl.Rule(profitability['Medium'] & risk['Medium'], decision['Neutral'])
rule3 = ctrl.Rule(profitability['Low'] | risk['High'], decision['Bad'])

system_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
system = ctrl.ControlSystemSimulation(system_ctrl)


system.input['profitability'] = input_profitability
system.input['risk'] = input_risk

system.compute()
decision_result = system.output["decision"]
print(f'Fuzzy decision: {decision_result:.2f}')

decision.view(sim=system)
plt.title(f'Result of Fuzzy Decision (Inputs: Profitability=7, Risk=3, Output={decision_result:.2f})')

plt.show()