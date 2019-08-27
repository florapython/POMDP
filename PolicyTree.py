# Flora Bouchacourt

import numpy, os, pdb, pylab

class PolicyTree:
    """ Policy tree """

    def __init__(self, spec, counter) : 
        self.specF={} #empty dic
        self.specF['top_node_action'] = spec.get('top_node_action',0) # action at the top node of the policy tree
        self.specF['list_of_observations_and_actions'] = spec.get('list_of_observations_and_actions',0)
        self.specF['value_of_the_policy_tree'] = spec.get('value_of_the_policy_tree',0) # values of states, length the number of states
        self.specF['horizon'] = spec.get('horizon',0)
        self.specF['successors'] = spec.get('successors',None) # subpolicies, themselves being a PolicyTree object. In the tiger problem it is a list of 2 elements, one for observation left, the other one for observation right
        self.id = counter
        self.key = str(self.specF['top_node_action'])+'_'+str(self.specF['horizon'])
        self.specF['optimal_alpha_map'] = spec.get('optimal_alpha_map',None) 

    def __hash__(self) :
        return self.id


    def __eq__(self,other) :
        return self.id==other.id

    def set_optimal_belief_values(self,belief_value) :
        self.specF['optimal_alpha_map'] = belief_value
        return

    def set_change_key(self) :
        self.key = str(self.specF['top_node_action'])
        return

