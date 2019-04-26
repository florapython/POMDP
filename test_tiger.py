# Flora Bouchacourt
# April 2019 : playing with the Tiger problem from Kaelbling 1998

import numpy, os
from Tiger import *

#figure 11 to 13
#MyTiger = Tiger({'discounting_parameter':1,'give_specific_initial_belief':False, 'name_to_save_folder':'Finite_horizon_tiger'})

#figure 14 to 16
MyTiger = Tiger({'discounting_parameter':0.95,'give_specific_initial_belief':False, 'name_to_save_folder':'Infinite_horizon_tiger'})
MyTiger.plot_plan_graph()

# figure 18
#MyTiger = Tiger({'discounting_parameter':0.95,'give_specific_initial_belief':False, 'name_to_save_folder':'Infinite_horizon_tiger_listen065', 'proba_of_correct_listening':0.65})
#MyTiger.compute_ValueIteration(horizon=200)
