# Flora Bouchacourt
# April 2019 : playing with the Tiger problem from Kaelbling 1998
# August 2019 : going back to this script, in order to have exhaustive value iteration, policy iteration and heuristic search working

import numpy, os
from Tiger import *
from Tiger_multiplecoherence import *
import time
from Tiger_false import *
from Tiger_Oct21 import *

# ------------------ KAEBLING 1998 ----------------------------------
#figure 11 to 13
#MyTiger = Tiger({'discounting_parameter':1,'give_specific_initial_belief':False, 'name_to_save_folder':'Finite_horizon_tiger'})

#figure 14 to 16
#MyTiger = Tiger({'discounting_parameter':0.95,'give_specific_initial_belief':False, 'name_to_save_folder':'Infinite_horizon_tiger'})
#MyTiger.compute_ValueIteration(horizon=110)
#MyTiger.plot_plan_graph(False)
# figure 17
#MyTiger.plot_plan_graph(True)



# figure 18
#MyTiger = Tiger({'discounting_parameter':0.95,'give_specific_initial_belief':True, 'name_to_save_folder':'Infinite_horizon_tiger_listen065_initial_belief', 'proba_of_correct_listening':0.65})
#MyTiger.compute_ValueIteration(horizon=100)
#MyTiger.plot_plan_graph(True)

# ------------------ HANSEN 1998 ----------------------------------
#MyTiger = Tiger({'discounting_parameter':0.95,'give_specific_initial_belief':False, 'name_to_save_folder':'Policy_iteration_tiger_may20_vé'})
#MyTiger.compute_PolicyIteration()

"""
# WAS THIS REALLY DONE ? TO BE CHECKED, BY LOOKING AT DIFFERENCES BETWEEN TIGER AND TIGER MULTIPLE COHERENCES
# -------- TIGER WITH MULTIPLE COHERENCES -------------------------
MyTiger = Tiger({'discounting_parameter':0.95,'give_specific_initial_belief':False, 'name_to_save_folder':'May18_test_075_newbounds', 'proba_of_correct_listening':0.75})
time1 = time.time()
#MyTiger.compute_ValueIteration(horizon=200)
time2 = time.time()
time_diff = time2 - time1
print("TIME IS "+str(time_diff))
MyTiger.plot_plan_graph(False)
MyTiger.plot_plan_graph(True)
"""


# AUGUST 2019
# 1) Test of the value iteration algorithm
"""
MyTiger = Tiger({'discounting_parameter':0.95,'give_specific_initial_belief':False, 'name_to_save_folder':'Aug2019_Infinite_horizon_tiger_listen065_precision0001_diff','proba_of_correct_listening':0.65})
MyTiger.compute_ValueIteration(horizon=200)
MyTiger.plot_plan_graph(False)
MyTiger.plot_plan_graph(True)
"""
"""
# 2) Test of the policy iteration algorithm
MyTiger = Tiger({'discounting_parameter':0.95,'give_specific_initial_belief':False, 'name_to_save_folder':'Aug2019_Policy_iteration_tiger2'})
MyTiger.compute_PolicyIteration()
# SOME BUGS TO FIX, CF NOTES IN NOTEBOOK
"""

# 3) Heuristic search on FSC
os.system('rm -rf Oct2019_Heuristic_search_tiger/*')
MyTiger = Tiger({'discounting_parameter':0.95, 'name_to_save_folder':'Oct2019_Heuristic_search_tiger'})
MyTiger.compute_HeuristicSearch()
#MyTiger.update_Belief_Heuristic_Search('TR',[1,0],'LEFT')

