# Flora Bouchacourt

import numpy, os, pdb, pylab

class Node:
    """ Node in a FSC """

    def __init__(self, spec, counter) : 
        self.specF={} #empty dic
        self.id = counter
        self.specF['action'] = spec.get('action','')
        #self.specF['alpha_vector'] = spec.get('alpha_vector',None)
        self.specF['successors'] = spec.get('successors',None) # id of nodes for the two observations, left right
        self.specF['value_of_states'] = spec.get('value_of_states',None)  # |S| 
        self.key = self.write_key()
        self.specF['node_in_alpha_max'] = spec.get('node_in_alpha_max',True)
        self.specF['node_belief'] = spec.get('node_belief',None) # |S|, in case the node represents only one belief state (heuristic search), left, right
        self.specF['lower_bound'] = spec.get('lower_bound',None)
        self.specF['previous_lower_bound'] = spec.get('previous_lower_bound',None)
        self.specF['upper_bound'] = spec.get('upper_bound',None)
        self.specF['reach_proba'] = spec.get('reach_proba',None)
        self.specF['depth'] = spec.get('depth',None)

    def __hash__(self) :
        return self.id


    def __eq__(self,other) :
        return self.id==other.id


    def write_key(self) :
        return str(self.id)+'_'+self.specF['action']


    def modify_node(self,new_action,new_successors,value_state) :
        self.specF['action'] = new_action
        #self.specF['alpha_vector'] = value_belief
        self.specF['successors'] = new_successors
        self.specF['value_of_states']  = value_state
        return

    def modify_node_with_belief(self,new_action,new_successors,value_state,node_belief) :
        self.specF['action'] = new_action
        self.specF['successors'] = new_successors
        self.specF['value_of_states']  = value_state
        self.specF['node_belief'] = node_belief
        return

    def update_lower_bound(self) :
        self.specF['previous_lower_bound'] = self.specF['lower_bound'].copy()
        self.specF['lower_bound'] = numpy.multiply(self.specF['node_belief'],specF['value_of_states'])
        return

    def update_upper_bound(self,value) :
        self.specF['upper_bound'] = value
        return
