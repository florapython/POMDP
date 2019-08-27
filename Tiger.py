# Flora Bouchacourt

import numpy, os, pdb, pylab, scipy
from matplotlib import pyplot as plt
from PolicyTree import *
#import networkx as nx
#import graphviz
#from networkx.drawing import *
#from networkx.drawing.nx_agraph import graphviz_layout
import pygraphviz as pgv

from FiniteStateController import *
from Node import *
from scipy.optimize import linprog


class Tiger:
    """ Class that runs a Tiger task episode.  """
    def __init__(self,spec):
        self.specF={} #empty dic
        self.specF['name_to_save_folder']=spec.get('name_to_save_folder','test') # folder name to save simulation results
        if not os.path.exists(self.specF['name_to_save_folder']+'/'):
            os.makedirs(self.specF['name_to_save_folder']+'/')
        self.specF['number_of_doors'] = spec.get('number_of_doors',2) # number of doors in the tiger problem
        if self.specF['number_of_doors'] != 2 :
            print("\nERROR : The class has to be rewritten for more than 2 doors")
            pdb.set_trace()
        self.specF['number_of_actions'] = spec.get('number_of_actions',3) # number of possible actions to do at each timestep
        self.specF['action_space'] = spec.get('action_space',numpy.asarray(['LEFT','RIGHT','LISTEN'])) # an array describing possible actions
        self.specF['state_space'] = spec.get('state_space',numpy.asarray(['LEFT','RIGHT'])) # an array describing the state space : where the tiger can be
        self.specF['possible_observations'] = spec.get('possible_observations',numpy.asarray(['TL','TR'])) # array describing possible observations
        self.specF['proba_of_correct_listening'] = spec.get('proba_of_correct_listening',0.85) # probability that the observation is correct ("coherence in Shadlen task")
        self.specF['proba_of_correct_observation_after_opening_a_door']  = spec.get('proba_of_correct_observation_after_opening_a_door',0.5) 
        self.specF['reward_from_listening'] = spec.get('reward_from_listening',-1)
        self.specF['reward_from_opening_the_correct_door'] = spec.get('reward_from_opening_the_correct_door',10)
        self.specF['reward_from_opening_the_wrong_door'] = spec.get('reward_from_opening_the_wrong_door',-100)
        self.specF['time_step_in_episode'] = spec.get('time_step_in_episode',0)
        #self.specF['belief_state'] = spec.get('belief_state',0.5) # belief that the tiger is behind the right door, the belief for state left is 1-belief
        self.specF['proba_reset_tiger'] = spec.get('proba_reset_tiger',0.5)
        self.specF['discounting_parameter'] = spec.get('discounting_parameter',1.0)
        if self.specF['discounting_parameter']>1 or self.specF['discounting_parameter']<0 :
            print("\nERROR : The discounting parameter takes value between 0 and 1")
            pdb.set_trace()
        self.specF['precision_parameter'] = spec.get('precision_parameter',0.0001) # 0.0001 in the Hansen thesis page 133
        self.specF['precision_parameter_beliefvector'] = spec.get('precision_parameter_beliefvector',0.0001)
        self.specF['belief_vector'] = spec.get('belief_vector',numpy.transpose(numpy.vstack((numpy.arange(0,1.+self.specF['precision_parameter_beliefvector']/2.,self.specF['precision_parameter_beliefvector']),1-numpy.arange(0,1.+self.specF['precision_parameter_beliefvector']/2.,self.specF['precision_parameter_beliefvector']))))) # size belief length * states
        self.specF['tiger_location'] = self.decide_NewTigerLocation()
        self.specF['residual_epsilon'] = spec.get('residual_epsilon',0.1)
        self.specF['bellman_residual'] = self.specF['residual_epsilon']*(1-self.specF['discounting_parameter'])/self.specF['discounting_parameter']
        self.specF['give_specific_initial_belief'] = spec.get('give_specific_initial_belief',False)
        self.specF['precision_pointwise_dominance'] = spec.get('precision_pointwise_dominance',0.01)
        self.specF['heuristic_search_error'] = spec.get('heuristic_search_error',0)
        self.specF['root_node_id'] = spec.get('root_node_id',0)
        self.specF['initial_belief_state'] = spec.get('initial_belief_state',numpy.asarray([0.5,0.5])) # Belief left, belief right

    def decide_NewTigerLocation(self) :
        p = numpy.random.rand()
        if p < self.specF['proba_reset_tiger'] :
            return self.specF['state_space'][0] # the tiger is behind the left door
        else :
            return self.specF['state_space'][1] # the tiger is behind the right door


    def reset_Episode(self) :
        self.specF['tiger_location'] = self.decide_NewTigerLocation()
        self.specF['time_step_in_episode'] = 0


    def run_ATrial(self) :
        action = self.make_AnAction()
        observation = self.observe_FromAction(action)
        reward = self.get_Reward(action)
        self.update_Belief(observation)
        self.specF['time_step_in_episode']+=1


    def make_AnAction(self) :
        """
        if :
            return "RIGHT"
        if :
            return "LEFT"
        if :
            return "LISTEN"
        """
        return

    def get_Reward(self,action) : # return the reward, and restart the episode if left or right are chosen
        if action=="LISTEN" :
            return self.specF['reward_from_listening']
        elif action==self.specF['tiger_location'] :
            self.reset_Episode()
            print("\nTIGER EATS YOU\n")
            return self.specF['reward_from_opening_the_wrong_door']
        else :
            self.reset_Episode()
            print("\nYOU WIN\n")
            return self.specF['reward_from_opening_the_correct_door']



    def observe_FromAction(self,action) :
        p = numpy.random.rand()
        if action=="LISTEN" :
            proba_to_compare = self.specF['proba_of_correct_listening'] 
        else :
            proba_to_compare = self.specF['proba_of_correct_observation_after_opening_a_door']  
        if p < proba_to_compare :
            return self.specF['tiger_location']
        else :
            return self.specF['state_space'][numpy.where(self.specF['state_space']!=self.specF['tiger_location'])[0][0]]


    def update_Belief(self,observation) : # Paragraph 3.3 of Kaelbling 1998 paper
        b_R_prior = self.specF['belief_state']
        b_L_prior = 1-self.specF['belief_state']
        if observation == "RIGHT" :
            proba_observation = self.specF['proba_of_correct_listening']*self.specF['belief_state'] + (1-self.specF['proba_of_correct_listening'])*(1-self.specF['belief_state'])
            b_R_posterior = self.specF['proba_of_correct_listening']*self.specF['belief_state']/proba_observation
        elif observation == "LEFT" :
            proba_observation = self.specF['proba_of_correct_listening']*(1-self.specF['belief_state']) + (1-self.specF['proba_of_correct_listening'])*self.specF['belief_state']
            b_R_posterior = (1-self.specF['proba_of_correct_listening'])*self.specF['belief_state']/proba_observation # b_R_posterior+b_L_posterior=1
        self.specF['belief_state']=b_R_posterior
        return 






    def compute_OptimalValue(self,Matrix, List_of_policies) :
        MaxValue = numpy.ones(self.specF['belief_vector'].shape[0])*numpy.nan
        List_index_optimal_policies = []
        optimal_alpha_map = {}
        for index, belief_value in enumerate(self.specF['belief_vector'][:,0]) :
            Value_for_that_belief_for_all_policies = Matrix[index,:]
            MaxValue[index] = numpy.amax(Value_for_that_belief_for_all_policies) # max over policies
            List_index = numpy.where(Value_for_that_belief_for_all_policies==MaxValue[index])[0]
            for index_policy in List_index :
                if index_policy in optimal_alpha_map :
                    optimal_alpha_map[index_policy].append(belief_value)
                else :
                    optimal_alpha_map[index_policy]=[belief_value]
                if index_policy not in List_index_optimal_policies :
                    List_index_optimal_policies.append(index_policy)
        List_to_prune = []

        for index_policy1 in list(optimal_alpha_map.keys()) :
            for index_policy2 in  list(optimal_alpha_map.keys()) :
                if index_policy1!=index_policy2 :
                    if index_policy1 not in List_to_prune and index_policy2 not in List_to_prune:
                        belief_set1 = optimal_alpha_map[index_policy1]
                        belief_set2 =  optimal_alpha_map[index_policy2]
                        #if index_policy1==4 and index_policy2==18 : pdb.set_trace()
                        if len(belief_set1)==len(belief_set2) and belief_set1==belief_set2: # the policies have exactly the same values, we need to keep only 1
                            """
                            if List_of_policies[index_policy1].specF['top_node_action']=='LISTEN' :
                                List_to_prune.append(index_policy2)
                            elif List_of_policies[index_policy2].specF['top_node_action']=='LISTEN' :
                                List_to_prune.append(index_policy1)
                            else :
                            """
                            List_to_prune.append(index_policy2)
                        elif all(elem in belief_set1  for elem in belief_set2) :
                            List_to_prune.append(index_policy2)
                        elif all(elem in belief_set2  for elem in belief_set1) :
                            List_to_prune.append(index_policy1)

        for index_to_prune in List_to_prune :
            List_index_optimal_policies.remove(index_to_prune)
        for index_optimal in List_index_optimal_policies :
            List_of_policies[index_optimal].set_optimal_belief_values(optimal_alpha_map[index_optimal])

        return MaxValue, List_index_optimal_policies



    def prune_Lark_algorithm(self, values, List_of_policies) : # Lark's algorithm for pruning, 1991, as it is written in Cassandra, Littman and Zhang 2013, values is a numpy array with dimension (self.specF['state_space'].shape[0],len(List_of_policies))
        List_index_optimal_policies = []
        # CREATING THE SET OF VECTORS F
        F = []
        for index_policy in range(values.shape[1]) :
            F.append(index_policy)
        if len(List_of_policies)==1 :
            List_index_optimal_policies = F.copy()
        else :
            # ADDING THE VECTORS MAXIMIZING THE EXTREMAS
            for index_state in range(self.specF['state_space'].shape[0]) :
                w = numpy.argmax(values[index_state,:])

                # test that extremas are well chosen
                max_value_here = numpy.amax(values[index_state,:]) 
                List_index = numpy.where(values[index_state,:]==max_value_here)[0]
                if List_index.shape[0] > 1 :
                    for index_policy_here in List_index :
                        for state_here in range(self.specF['state_space'].shape[0]):
                            if values[state_here,index_policy_here] > values[state_here,w] :
                                print("PROBLEM IS CHOSSING EXTREMAS VECTORS IN PRUNING")
                                pdb.set_trace()


                if w not in List_index_optimal_policies :
                    List_index_optimal_policies.append(w)
                F.remove(w)
                #print("end states")
            
            while F :
                #print(F)
                Phi = F[0]
                #print("Phi is "+str(Phi))
                successLP, dominated, vector_x = self.Dominate(Phi, List_index_optimal_policies, values)
                if successLP == False :
                    F.remove(Phi)
                elif dominated :
                    F.remove(Phi)
                else :
                    max_val = -numpy.inf
                    best = None
                    for w in F :
                        val = numpy.dot(vector_x,values[:,w])
                        if val > max_val :
                            max_val = val
                            best = w
                    if best not in List_index_optimal_policies :
                        redondance = self.check_redondance(best,List_index_optimal_policies,values) 
                        if redondance==False :
                            List_index_optimal_policies.append(best)
                    F.remove(best)

        optimal_alpha_map = {}
        MaxValue = numpy.ones(self.specF['belief_vector'].shape[0])*numpy.nan
        Values_belief = numpy.dot(self.specF['belief_vector'],values)
        for index, belief_value in enumerate(self.specF['belief_vector'][:,0]) :
            MaxValue[index] = numpy.amax(Values_belief[index,:]) # max over policies
            List_index = numpy.where(Values_belief[index,:]==MaxValue[index])[0]
            for index_policy in List_index :
                if index_policy in List_index_optimal_policies :
                    if index_policy in optimal_alpha_map :
                        optimal_alpha_map[index_policy].append(belief_value)
                    else :
                        optimal_alpha_map[index_policy]=[belief_value]

        for index_optimal in List_index_optimal_policies :
            try :
                List_of_policies[index_optimal].set_optimal_belief_values(optimal_alpha_map[index_optimal])
            except :
                print("optimal policy "+str(index_optimal)+" has no optimal map")
                continue
        
        return MaxValue, List_index_optimal_policies


    def check_redondance(self,best_to_check, List_index_optimal_policies,values) :
        for index_policy in List_index_optimal_policies :
            if numpy.allclose(values[:,index_policy], values[:,best_to_check],rtol=self.specF['precision_parameter'],atol=self.specF['precision_parameter']):
                return True
            elif values[0,index_policy]==values[0,best_to_check] and values[1,index_policy] > values[1,best_to_check] :
                return True
            elif values[1,index_policy]==values[1,best_to_check] and values[0,index_policy] > values[0,best_to_check] :
                return True
            else :
                return False


    def Dominate(self,alpha,A, val) : # Linear programming returns a belief state value for which alpha gives a larger dot product than any vector in A
    # Description in Figure2 of Cassandra et al 2013 and in paragraph 2.4 in http://www.bgu.ac.il/~shanigu/Publications/skyline_AMAI.3.pdf for better understanding
    # 3 variables x(s1), x(s2) and -delta
    # REFERENCE : https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.linprog.html
        dominated=False
        #successLP = True
        
        # A_eq and b_eq implement x.1=1
        A_eq = numpy.array([numpy.append(numpy.ones(self.specF['state_space'].shape[0]), [0.])]) # [[1., 1., 0.]]
        b_eq = numpy.array([1.]) # [1]
        # instead of maximizing delta, we will minimize -delta, as the linprog function of scipy is a minimizer
        c = numpy.append(numpy.zeros(self.specF['state_space'].shape[0]), [1.]) # [0., 0., 1.], permits to minimize -delta (delta 3rd variable in LP)
        
        for index, vector_w in enumerate(A) :

            if index==0 :
                A_ub  = numpy.array([numpy.append(-val[:,alpha]+val[:,vector_w], [-1.])])
                b_ub = numpy.array([-self.specF['precision_parameter']])
            else :           
                A_ub = numpy.vstack((A_ub ,(numpy.array([numpy.append(-val[:,alpha]+val[:,vector_w], [-1.])])))) # coefficients for [[x1,x2,-delta]]
                b_ub = numpy.vstack((b_ub, numpy.array([-self.specF['precision_parameter']])))

        #results = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, None))
        results = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=[(0,None),(0,None),(None,None)],method='interior-point')
        successLP = results.success
        if successLP :
            if results.x[self.specF['state_space'].shape[0]]>0 : # results.x[self.specF['state_space'].shape[0]] is -delta :
                dominated=True
                final_vector=None
            else :
                dominated=False
                final_vector = results.x[:self.specF['state_space'].shape[0]]
        else :
            dominated=True
            final_vector=None
        
        """
        # THIS IS WRONG
        for index, vector_w in enumerate(A) :
            A_ub  = numpy.array([numpy.append(-val[:,alpha]+val[:,vector_w], [-1.])])
            b_ub = numpy.array([-self.specF['precision_parameter']])
            results = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, None))
            successLP = results.success
            if successLP :
                if results.x[self.specF['state_space'].shape[0]]>0 : # results.x[self.specF['state_space'].shape[0]] is -delta :
                    dominated=True
                    final_vector=None
                    break
                else :
                    final_vector = results.x[:self.specF['state_space'].shape[0]] 
            else :
                dominated=True
                final_vector=None
                break
        """
        return successLP, dominated, final_vector

    def plot_graph(self, time_step,DG,label_dict,edge_label_dict) :
        obs_left = [(u, v) for (u, v, d) in DG.edges(data=True) if d['weight']=='TL']
        obs_right = [(u, v) for (u, v, d) in DG.edges(data=True) if d['weight']=='TR']
        plt.figure()
        pos = graphviz_layout(DG)
        nx.draw(DG, with_labels=True, pos=pos, node_size=3500, node_color='white', edgecolors='black',labels=label_dict)
        #pdb.set_trace()
        #nx.draw_networkx_edge_labels(DG,pos,font_size=8,edge_label=DG.edges(data='weight'))
        nx.draw_networkx_edge_labels(DG,pos,font_size=8,edge_label=edge_label_dict)
        #nx.draw_networkx_nodes(DG, with_labels=True, pos=pos, node_size=3500, node_color='white', labels=label_dict)
        #nx.draw_networkx_edges(DG, pos, edgelist=obs_left,edge_color='r')
        #nx.draw_networkx_edges(DG, pos, edgelist=obs_right,edge_color='b')
        plt.savefig(self.specF['name_to_save_folder']+'/policy_graph_'+str(time_step))
        return


    def test_equal_machine_state(self,action,successors,FSC) :
        equality = False
        nodeid = None
        for node in FSC.specF['list_of_nodes'] :
            if action==node.specF['action'] and numpy.sum(numpy.equal(successors,node.specF['successors']))==self.specF['possible_observations'].shape[0] :
                equality=True
                nodeid = node.id
                break
        return equality, nodeid

    
    def test_equal_machine_state_heuristic(self,new_node, Previous_delta) :
        equality = False
        nodeid = None
        action = new_node.specF['action']
        successors = new_node.specF['successors']
        for node in Previous_delta :
            if action==node.specF['action'] and numpy.sum(numpy.equal(successors,node.specF['successors']))==self.specF['possible_observations'].shape[0] :
                equality=True
                nodeid = node.id
                break
        return equality, nodeid




    def test_pointwise_dominance(self,new_value, FSC) : 
        list_of_domination = []
        """
        # on belief states
        value_belief = numpy.dot(self.specF['belief_vector'],new_value)
        for node in FSC.specF['list_of_nodes'] :
            alpha_vector = numpy.dot(self.specF['belief_vector'],node.specF['value_of_states'])
            if numpy.sum(numpy.less_equal(alpha_vector+numpy.ones(alpha_vector.shape[0])*self.specF['precision_pointwise_dominance'],value_belief)) == value_belief.shape[0] :
                list_of_domination.append(node.id)
        """
        # TODO on states only ?
        for node in FSC.specF['list_of_nodes'] :
            if numpy.sum(numpy.less_equal(node.specF['value_of_states']+numpy.ones(node.specF['value_of_states'].shape[0])*self.specF['precision_pointwise_dominance'],new_value))== new_value.shape[0] :
                list_of_domination.append(node.id)
        return list_of_domination





    def prune_vectors_from_DP_on_FSC(self,currentFSC) : # old pruning method before implementing Lark algo
        Matrix_value = numpy.ones((self.specF['belief_vector'].shape[0],len(currentFSC.specF['list_of_nodes'])))*numpy.nan
        MaxValue = numpy.ones(self.specF['belief_vector'].shape[0])*numpy.nan
        optimal_alpha_map = {}
        for node_z in currentFSC.specF['list_of_nodes'] :
            index = currentFSC.specF['list_of_nodes'].index(node_z)
            Matrix_value[:,index] = numpy.dot(self.specF['belief_vector'],node_z.specF['value_of_states'])
        List_index_optimal_policies = []
        for index, belief_value in enumerate(self.specF['belief_vector'][:,0]) :
            Value_for_that_belief_for_all_policies = Matrix_value[index,:]
            MaxValue[index] = numpy.amax(Value_for_that_belief_for_all_policies) # max over policies
            List_index = numpy.where(Value_for_that_belief_for_all_policies==MaxValue[index])[0]
            for index_policy in List_index :
                if index_policy in optimal_alpha_map :
                    optimal_alpha_map[index_policy].append(belief_value)
                else :
                    optimal_alpha_map[index_policy]=[belief_value]
                if index_policy not in List_index_optimal_policies :
                    List_index_optimal_policies.append(index_policy)
        List_to_prune = []

        for index_policy1 in list(optimal_alpha_map.keys()) :
            for index_policy2 in  list(optimal_alpha_map.keys()) :
                if index_policy1!=index_policy2 :
                    if index_policy1 not in List_to_prune and index_policy2 not in List_to_prune:
                        belief_set1 = optimal_alpha_map[index_policy1]
                        belief_set2 =  optimal_alpha_map[index_policy2]
                        if len(belief_set1)==len(belief_set2) and belief_set1==belief_set2: # the policies have exactly the same values, we need to keep only 1
                            node_policy1 = currentFSC.specF['list_of_nodes'][index_policy1]
                            node_policy2 = currentFSC.specF['list_of_nodes'][index_policy2]
                            
                            if node_policy1.specF['action']=='LISTEN' :
                                List_to_prune.append(index_policy2)
                            elif node_policy2.specF['action']=='LISTEN' :
                                List_to_prune.append(index_policy1)
                            else :
                                List_to_prune.append(index_policy2)
                            
                    if (index_policy2 not in List_to_prune) and all(elem in belief_set1  for elem in belief_set2) :
                        List_to_prune.append(index_policy2)
                    elif (index_policy1 not in List_to_prune) and all(elem in belief_set2  for elem in belief_set1) :
                        List_to_prune.append(index_policy1)
        for index_to_prune in List_to_prune :
            List_index_optimal_policies.remove(index_to_prune)
        for node_z in currentFSC.specF['list_of_nodes'] :
            index = currentFSC.specF['list_of_nodes'].index(node_z)
            if index in List_index_optimal_policies :
               node_z.specF['node_in_alpha_max'] = True
            else :
              node_z.specF['node_in_alpha_max'] = False
        return MaxValue, List_index_optimal_policies, optimal_alpha_map, currentFSC

    def prune_Lark_algorithm_on_FSC(self,currentFSC) : # same as prune_Lark_algorithm but for FSC with policy iteration
        currentFSC.update_value_of_states() # maybe that's unecessary at this point
        values = currentFSC.specF['value_of_states']
        List_index_optimal_policies = []
        # CREATING THE SET OF VECTORS F
        F = []
        for index_policy in range(values.shape[1]) :
            F.append(index_policy)
        # ADDING THE VECTORS MAXIMIZING THE EXTREMAS
        for index_state in range(self.specF['state_space'].shape[0]) :
            w = numpy.argmax(values[index_state,:])

            # test that extremas are well chosen
            max_value_here = numpy.amax(values[index_state,:]) 
            List_index = numpy.where(values[index_state,:]==max_value_here)[0]
            if List_index.shape[0] > 1 :
                for index_policy_here in List_index :
                    for state_here in range(self.specF['state_space'].shape[0]):
                        if values[state_here,index_policy_here] > values[state_here,w] :
                            print("PROBLEM IS CHOOSING EXTREMAS VECTORS IN PRUNING")
                            pdb.set_trace()


            if w not in List_index_optimal_policies :
                List_index_optimal_policies.append(w)
            F.remove(w)
            #print("end states")
        
        while F :
            #print(F)
            Phi = F[0]
            #print("Phi is "+str(Phi))
            successLP, dominated, vector_x = self.Dominate(Phi, List_index_optimal_policies, values)
            if successLP == False :
                F.remove(Phi)
            elif dominated :
                F.remove(Phi)
            else :
                max_val = -numpy.inf
                best = None
                for w in F :
                    val = numpy.dot(vector_x,values[:,w])
                    if val > max_val :
                        max_val = val
                        best = w
                if best not in List_index_optimal_policies :
                    redondance = self.check_redondance(best,List_index_optimal_policies,values) 
                    if redondance==False :
                        List_index_optimal_policies.append(best)
                F.remove(best)

        for node_z in currentFSC.specF['list_of_nodes'] :
            index = currentFSC.specF['list_of_nodes'].index(node_z)
            if index in List_index_optimal_policies :
               node_z.specF['node_in_alpha_max'] = True
            else :
              node_z.specF['node_in_alpha_max'] = False
        return List_index_optimal_policies, currentFSC


    def delete_nodes(self,currentFSC) :
        list_to_prune_id = []
        for node_z in currentFSC.specF['list_of_nodes'] :
            if node_z.specF['node_in_alpha_max'] == False :
                list_to_prune_id.append(node_z.id)
        if len(list_to_prune_id)!=0 :
            for index_to_prune in list_to_prune_id :
                node_to_delete = currentFSC.find_node_from_nodeid(index_to_prune)
                currentFSC.remove_node(currentFSC.specF['list_of_nodes'][node_to_delete])
        currentFSC.update_value_of_states()
        return currentFSC


    def check_if_node_is_reached_by_other_nodes(self,nodeid, List) :
        result = False
        for node in List :
            if nodeid in node.specF['successors'] :
                result = True
                break
        return result

    def estimate_value_of_node(self,index_action,node1,node2,observation_matrix,transition_matrix,reward_matrix) :
        sum_over_observations = numpy.multiply(observation_matrix[index_action,:,0],node1.specF['value_of_states']) + numpy.multiply(observation_matrix[index_action,:,1],node2.specF['value_of_states'])
        sum_over_transitions = numpy.dot(transition_matrix[index_action,:,:],sum_over_observations) # dim 2*1
        new_values = reward_matrix[index_action]+self.specF['discounting_parameter']*sum_over_transitions # dim 2*1
        return new_values





    def compute_PolicyIteration(self) : # Hansen 1998
        step_of_policy_iteration = 0
        transition_matrix = self.get_transition_matrix()
        reward_matrix = self.get_reward_matrix()
        observation_matrix = self.get_observation_matrix()
        terminate = False
        List_residual = []
        # INITIALISATION : specify an initial FSC, the initial FSC I choose is going to be constituted of each action as nodes, with connectivity on themselves
        MyFSC = FiniteStateController({'belief_vector':self.specF['belief_vector'],'possible_observations': self.specF['possible_observations']})
        # Policy evaluation of the initial state |S| x |nodes|
        index_action=2
        new_action = 'LISTEN'
        successors=numpy.asarray([0,0])
        sum_over_observations = numpy.multiply(observation_matrix[index_action,:,0],reward_matrix[index_action,:]) + numpy.multiply(observation_matrix[index_action,:,1],reward_matrix[index_action,:])
        sum_over_transitions = numpy.dot(transition_matrix[index_action,:,:],sum_over_observations) # dim 2*1
        new_values = reward_matrix[index_action]+self.specF['discounting_parameter']*sum_over_transitions # dim 2*1
        new_node_id = MyFSC.create_new_node(new_action, successors, new_values)

        MyFSC.plot(self.specF['name_to_save_folder'],step_of_policy_iteration) 
        MyFSC.update_value_of_states()
        value_of_belief = numpy.dot(self.specF['belief_vector'],MyFSC.specF['value_of_states']) # size of belief length x |nodes|
        alpha_max = self.compute_max_of_alpha_vectors(value_of_belief)
        self.plot_values(value_of_belief,alpha_max,step_of_policy_iteration)
        List_residual.append(1)
        recompute_policy_eval = False

        while terminate==False :
            print('\n--------- ITERATION STEP ---------'+str(step_of_policy_iteration))
            alpha_max_previous = alpha_max.copy()

            # Policy improvement
            recompute_policy_eval = False
            if len(MyFSC.specF['value_of_states'].shape) == 1 :
                if len(MyFSC.specF['list_of_nodes'])!= 1 : pdb.set_trace()
            else :
                if len(MyFSC.specF['list_of_nodes'])!=MyFSC.specF['value_of_states'].shape[1] : pdb.set_trace()

            # DYNAMIC PROGRAMMING
            number_of_backups = 0
            # Creation of a new FSC
            TransientFSC = FiniteStateController({'belief_vector':self.specF['belief_vector'],'possible_observations': self.specF['possible_observations']})
            for index_action, new_action in enumerate(self.specF['action_space']) :
                for node1 in MyFSC.specF['list_of_nodes'] :
                    if node1.specF['node_in_alpha_max'] :
                        for node2 in MyFSC.specF['list_of_nodes'] :
                            if node2.specF['node_in_alpha_max'] :
                                new_values  = self.estimate_value_of_node(index_action,node1,node2,observation_matrix,transition_matrix,reward_matrix)
                                new_node_id = TransientFSC.create_new_node(new_action, numpy.asarray([node1.id, node2.id]), new_values)
                                number_of_backups+=1
                            """
                            else :
                                print("non optimal sub FSC ?")
                                pdb.set_trace()
                            """
                    """
                    else :
                        print("non optimal sub FSC ?")
                        pdb.set_trace()
                    """

            #MaxValue, List_index_optimal_policies, optimal_alpha_map, TransientFSC = self.prune_vectors_from_DP_on_FSC(TransientFSC)
            print("Pruning the transient FSC...")
            List_index_optimal_policies, TransientFSC = self.prune_Lark_algorithm_on_FSC(TransientFSC)
            TransientFSC = self.delete_nodes(TransientFSC)

            List_of_new_nodeid = []
            List_of_node_to_merge = []
            # for each vector in TransientFSC (V', in Hansen)
            for new_node_fsc in TransientFSC.specF['list_of_nodes'] :
                #if new_node_fsc.specF['node_in_alpha_max']==False : pdb.set_trace()
                if new_node_fsc.specF['node_in_alpha_max'] :
                    # Test if the action and the successor links associated with it are the same as those of a machine state already in MyFSC (Delta in Hansen)
                    test1, nodeid = self.test_equal_machine_state(new_node_fsc.specF['action'],new_node_fsc.specF['successors'],MyFSC)
                    if test1 :
                        List_of_new_nodeid.append(nodeid)
                    else :
                        # Else test if the vector pointwise dominates a vector associated with a machine state in MyFSC
                        liste_of_domination_test2 = self.test_pointwise_dominance(new_node_fsc.specF['value_of_states'],MyFSC)
                        if len(liste_of_domination_test2)!=0 : # a new vector dominates old vectors
                            recompute_policy_eval=True
                            if len(liste_of_domination_test2)==1 : # only one old vector is dominated, the old vector will be changed
                                nodeid = liste_of_domination_test2[0]
                                MyFSC.change_action_and_successors(nodeid,new_node_fsc.specF['action'],new_node_fsc.specF['successors'],new_node_fsc.specF['value_of_states'])
                                List_of_new_nodeid.append(nodeid)
                            else : # more than one old vector is dominated, one old vector will be changed, and the other one deleted
                                selected_node=False
                                for other_node_id in liste_of_domination_test2 :
                                    flattened_ids = numpy.asarray(List_of_node_to_merge.copy())
                                    flattened_ids = flattened_ids.flatten()
                                    if (other_node_id not in flattened_ids) and selected_node==True:
                                        List_of_node_to_merge.append([other_node_id,nodeid])
                                    if (other_node_id not in flattened_ids) and selected_node==False:
                                        nodeid = other_node_id
                                        MyFSC.change_action_and_successors(nodeid,new_node_fsc.specF['action'],new_node_fsc.specF['successors'],new_node_fsc.specF['value_of_states'])
                                        List_of_new_nodeid.append(nodeid)
                                        selected_node=True
                                if selected_node==False :
                                    print("PROBLEM SELECTED NODE")
                                    pdb.set_trace()

        
                        else : # no domination at all, we add this new machine state
                            new_node_id = MyFSC.create_new_node(new_node_fsc.specF['action'],new_node_fsc.specF['successors'],new_node_fsc.specF['value_of_states'])
                            List_of_new_nodeid.append(new_node_id)

            # Merging the dominated machine states
            if len(List_of_node_to_merge)!=0 :
                for tuple_merging in List_of_node_to_merge :
                    other_node_id = tuple_merging[0]
                    nodeid_merged = tuple_merging[1]
                    isreachedbyothernode = self.check_if_node_is_reached_by_other_nodes(other_node_id, MyFSC.specF['list_of_nodes'])
                    if isreachedbyothernode :
                        for index_node in MyFSC.specF['list_of_nodes'] :
                            for index_succ in range(index_node.specF['successors'].shape[0]) :
                                if other_node_id==index_node.specF['successors'][index_succ] :
                                    index_node.specF['successors'][index_succ]=nodeid_merged

                    node_to_delete = MyFSC.find_node_from_nodeid(other_node_id)
                    MyFSC.remove_node(MyFSC.specF['list_of_nodes'][node_to_delete])

            # Pruning machine states from previous fsc that are not reachable anymore from any initial belief value
            for node in MyFSC.specF['list_of_nodes'] :
                if node.id in List_of_new_nodeid :
                    continue
                else :
                    isreachedbyothernode = self.check_if_node_is_reached_by_other_nodes(node.id, MyFSC.specF['list_of_nodes'])
                    if isreachedbyothernode :
                        continue
                    else : # delete node
                        MyFSC.remove_node(node)

            MyFSC.update_value_of_states()
            alpha_max_comp = numpy.dot(self.specF['belief_vector'],MyFSC.specF['value_of_states'])
            alpha_max = self.compute_max_of_alpha_vectors(alpha_max_comp)
            step_of_policy_iteration+=1
            MyFSC.plot(self.specF['name_to_save_folder'],step_of_policy_iteration)   
            self.plot_values(alpha_max_comp,alpha_max,step_of_policy_iteration)
            del TransientFSC

            # TEST TERMINATION
            vector_difference = numpy.absolute(alpha_max-alpha_max_previous)
            max_residual = numpy.amax(vector_difference)
            print('Residual is '+str(max_residual)+' To be compared to '+str(self.specF['bellman_residual']))
            List_residual.append(max_residual)
            if max_residual<self.specF['bellman_residual'] :
                terminate = True
                print("POLICY ITERATION TERMINATED AT STEP "+str(step_of_policy_iteration))

            # IS THIS POLICY EVALUATION ?
            value_has_converged = numpy.zeros(len(MyFSC.specF['list_of_nodes']),dtype=bool)
            while numpy.all(value_has_converged)==False:
                for nodes_in_fsc in MyFSC.specF['list_of_nodes'] :
                    index_node = MyFSC.specF['list_of_nodes'].index(nodes_in_fsc)
                    if value_has_converged[index_node]==False :
                        old_value = nodes_in_fsc.specF['value_of_states']
                        index_action = numpy.where(nodes_in_fsc.specF['action']==self.specF['action_space'])[0][0]
                        node1 = MyFSC.specF['list_of_nodes'][MyFSC.find_node_from_nodeid(nodes_in_fsc.specF['successors'][0])]
                        node2 = MyFSC.specF['list_of_nodes'][MyFSC.find_node_from_nodeid(nodes_in_fsc.specF['successors'][1])]
                        new_values = self.estimate_value_of_node(index_action,node1,node2,observation_matrix,transition_matrix,reward_matrix)
                        if numpy.sum(numpy.equal(new_values,old_value))!=new_values.shape[0] : # the value changed
                            value_has_converged[index_node] = False
                            nodes_in_fsc.specF['value_of_states'] = new_values
                        else :
                            value_has_converged[index_node] = True

            MyFSC.update_value_of_states()
            #MaxValue, List_index_optimal_policies, optimal_alpha_map, MyFSC = self.prune_vectors_from_DP_on_FSC(MyFSC)
            List_index_optimal_policies, MyFSC = self.prune_Lark_algorithm_on_FSC(MyFSC)
            # Here, don't delete nodes that are not in alpha max, they could be reachable from a machine state in alpha max, they should then be kept in the FSC

            print("FINAL NUMBER OF NODES IN THE FSC AT THIS STEP : "+str(MyFSC.specF['value_of_states'].shape[1]))
            
        self.plot_bellman_residual(List_residual)
        
        return



    def compute_max_of_alpha_vectors(self,value_of_belief) :
        alpha_max_fct = numpy.ones(self.specF['belief_vector'].shape[0])*numpy.nan
        if len(value_of_belief.shape)==1 : #case where there is only 1 node
            return value_of_belief
        else :
            for index_belief, belief_value in enumerate(self.specF['belief_vector'][:,0]) :
                alpha_max_fct[index_belief] = numpy.amax(value_of_belief[index_belief,:])
        return alpha_max_fct

    def plot_values(self,value_of_belief,alpha_max_fct,iteration_step) :
        plt.figure(figsize=(10,5))
        plt.subplot(121)
        if len(value_of_belief.shape)==1 : #case where there is only 1 node
            plt.plot(self.specF['belief_vector'][:,0],value_of_belief,label='policy')
        else :
            for index in range(value_of_belief.shape[1]) : # number of nodes
                plt.plot(self.specF['belief_vector'][:,0],value_of_belief[:,index],label='policy '+str(index))
        plt.legend()
        plt.ylim(-30,10)
        plt.xlabel("belief left")
        plt.ylabel(str(iteration_step)+" step value function")
        plt.subplot(122)
        plt.plot(self.specF['belief_vector'][:,0],alpha_max_fct,'k:',label='optimal value')
        plt.xlabel("belief left")
        plt.ylabel(str(iteration_step)+" step policy tree value function")
        plt.savefig(self.specF['name_to_save_folder']+'/fig_'+str(iteration_step)+'step_alphamax.png')
        plt.close()
        return


    def compute_ValueIteration(self,horizon) : # Kaelbling 1998
        counter=0 # a running variable used in order to have an id for each PolicyTree objects
        terminate = False # test for terminating the iteration with bellman residual
        transition_matrix = self.get_transition_matrix()
        reward_matrix = self.get_reward_matrix()
        observation_matrix = self.get_observation_matrix()
        G=pgv.AGraph(strict=False,directed=True) # graph object, directed graph with possible parallel edges and self-loops
        List_residual = [] # residuals of the bellman residual test, to check for convergence
        for time_step in range(horizon) :
            List_of_policies = [] # list of objects from the class PolicyTree
            if time_step==0:
                if self.specF['give_specific_initial_belief'] : # the case where we know the initial belief state, e.g. fig 17 of the Kaelbling
                    initial_belief = self.get_initial_belief_state()
                    value_of_belief =  initial_belief[0]*reward_matrix[:,0]+initial_belief[1]*reward_matrix[:,1]
                    if value_of_belief.shape[0]!=self.specF['action_space'].shape[0] :
                        pdb.set_trace()
                    initial_action = numpy.argmax(value_of_belief)
                    MyNewPolicy = PolicyTree({'top_node_action':self.specF['action_space'][initial_action],'list_of_observations_and_actions': [self.specF['action_space'][initial_action]], 'value_of_the_policy_tree':reward_matrix[initial_action,:], 'horizon':time_step},counter)
                    counter+=1
                    List_of_policies.append(MyNewPolicy)
                else :
                    for index, new_action in enumerate(self.specF['action_space']) :
                        MyNewPolicy = PolicyTree({'top_node_action':new_action,'list_of_observations_and_actions': [new_action], 'value_of_the_policy_tree':reward_matrix[index], 'horizon':time_step},counter)
                        counter+=1
                        List_of_policies.append(MyNewPolicy)
            else :
                # List_of_old_policies is the pruned list from the previous timestep, so the optimal policies at t-1
                for index_action, new_action in enumerate(self.specF['action_space']) : # action at the top of the new policy tree
                    for index_subpolicy1, subpolicy1 in enumerate(List_of_old_policies) : # subpolicy for observation left
                        for index_subpolicy2, subpolicy2 in enumerate(List_of_old_policies) : # subpolicy for observation right
                            List_of_current_subpolicies = [subpolicy1,subpolicy2]
                            # Creating the new list of observations and actions
                            NewList = []
                            for index_observation in range(self.specF['possible_observations'].shape[0]) :
                                old_policy = List_of_current_subpolicies[index_observation].specF['list_of_observations_and_actions'].copy()
                                old_policy.insert(0,self.specF['possible_observations'][index_observation]) # The insert() method inserts an element to the list at a given index
                                old_policy.insert(0,new_action)
                                NewList.append(old_policy)
                            # Value iteration step one : sum over observations
                            sum_over_observations = numpy.zeros(self.specF['state_space'].shape[0]) # s'
                            for index_observation in range(self.specF['possible_observations'].shape[0]) :
                                value_of_old_policy = List_of_current_subpolicies[index_observation].specF['value_of_the_policy_tree']
                                sum_over_observations += numpy.multiply(observation_matrix[index_action,:,index_observation],value_of_old_policy)
                            # Value iterations step two : sum over state transitions
                            sum_over_transitions = numpy.dot(transition_matrix[index_action,:,:],sum_over_observations) # dim 2*1
                            if sum_over_transitions.shape[0]!=self.specF['state_space'].shape[0] : pdb.set_trace()
                            # Computation of value of the policy tree
                            new_value_for_new_policy = reward_matrix[index_action]+self.specF['discounting_parameter']*sum_over_transitions # dim 2*1
                            if new_value_for_new_policy.shape[0]!=self.specF['state_space'].shape[0] : pdb.set_trace()

                            MyNewPolicy = PolicyTree({'top_node_action':new_action,'list_of_observations_and_actions': NewList, 'value_of_the_policy_tree':new_value_for_new_policy, 'horizon':time_step, 'successors':List_of_current_subpolicies},counter)
                            counter+=1
                            List_of_policies.append(MyNewPolicy)
                            

            print("HORIZON "+str(time_step+1))

            values = numpy.ones((self.specF['state_space'].shape[0],len(List_of_policies)))*numpy.nan
            for index_policy_tree in range(len(List_of_policies)) :
                values[:,index_policy_tree] = List_of_policies[index_policy_tree].specF['value_of_the_policy_tree']

            value_of_belief = numpy.dot(self.specF['belief_vector'],values)

            #alpha_max,List_index_optimal_policies = self.compute_OptimalValue(value_of_belief,List_of_policies)
            alpha_max, List_index_optimal_policies = self.prune_Lark_algorithm(values,List_of_policies)
            print(str(len(List_index_optimal_policies))+" OPTIMAL POLICIES")
            """
            alpha_max = numpy.ones(self.specF['belief_vector'].shape[0])*numpy.nan
            val_belief_optimal_policies = value_of_belief[:,List_index_optimal_policies]
            for index_belief_val in range(self.specF['belief_vector'].shape[0]) :
                alpha_max[index_belief_val] = numpy.amax(val_belief_optimal_policies[index_belief_val,:])
            """

            Pruned_policies = [] # it's a list of PolicyTree class instance
            for index, index_pruned_policies in enumerate(List_index_optimal_policies) :
                Pruned_policies.append(List_of_policies[index_pruned_policies])

            for index in range(len(Pruned_policies)) :
                if time_step==0 :
                    G.add_node(Pruned_policies[index].id)
                    n=G.get_node(Pruned_policies[index].id)
                    n.attr['label'] = Pruned_policies[index].key
                    n.attr['pos']="%f,%f)"%(index,time_step)
                if time_step > 0 :
                    if Pruned_policies[index].specF['successors'][0].id==Pruned_policies[index].specF['successors'][1].id :
                        G.add_edge(Pruned_policies[index].id,Pruned_policies[index].specF['successors'][0].id)
                        n=G.get_node(Pruned_policies[index].id)
                        n.attr['label'] = Pruned_policies[index].key
                        n.attr['pos']="%f,%f)"%(index,time_step)
                        nsucc0=G.get_node(Pruned_policies[index].specF['successors'][0].id)
                        nsucc0.attr['label'] = Pruned_policies[index].specF['successors'][0].key  
                        nsucc0.attr['pos']="%f,%f)"%(index,time_step-1)
                        e=G.get_edge(Pruned_policies[index].id,Pruned_policies[index].specF['successors'][0].id)
                        e.attr['label']='TL/TR'
                    else :
                        G.add_edge(Pruned_policies[index].id,Pruned_policies[index].specF['successors'][0].id)
                        G.add_edge(Pruned_policies[index].id,Pruned_policies[index].specF['successors'][1].id)
                        n=G.get_node(Pruned_policies[index].id)
                        n.attr['label'] = Pruned_policies[index].key
                        n.attr['pos']="%f,%f)"%(index,time_step)
                        nsucc0=G.get_node(Pruned_policies[index].specF['successors'][0].id)
                        nsucc0.attr['label'] = Pruned_policies[index].specF['successors'][0].key  
                        nsucc0.attr['pos']="%f,%f)"%(index-0.3,time_step-1)
                        nsucc1=G.get_node(Pruned_policies[index].specF['successors'][1].id)
                        nsucc1.attr['label'] = Pruned_policies[index].specF['successors'][1].key 
                        nsucc1.attr['pos']="%f,%f)"%(index+0.3,time_step-1)
                        e0=G.get_edge(Pruned_policies[index].id,Pruned_policies[index].specF['successors'][0].id)
                        e1=G.get_edge(Pruned_policies[index].id,Pruned_policies[index].specF['successors'][1].id)
                        e0.attr['label']=Pruned_policies[index].specF['list_of_observations_and_actions'][0][1]
                        e1.attr['label']=Pruned_policies[index].specF['list_of_observations_and_actions'][1][1]
            if time_step <= 5 or time_step > horizon-10:
                G.draw(self.specF['name_to_save_folder']+'/policy_graph_'+str(time_step)+'.png',prog='dot')

            # testing bellman residual
            if self.specF['discounting_parameter'] < 1 :
                if time_step==0 :
                    List_residual.append(numpy.amax(alpha_max))
                if time_step>0 :
                    vector_difference = numpy.absolute(alpha_max-alpha_max_previous)
                    max_residual = numpy.amax(vector_difference)
                    print('Residual is '+str(max_residual)+' To be compared to '+str(self.specF['bellman_residual']))
                    List_residual.append(max_residual)
                    if max_residual<self.specF['bellman_residual'] :
                        terminate = True

            # Alpha vector that will be used for testing the bellman residual at the next time step
            alpha_max_previous = alpha_max.copy()
            # Policies that will be used for the next time step
            List_of_old_policies = Pruned_policies.copy()
            
            if time_step <= 5 or time_step > horizon-10 :
                plt.figure()
                for index in List_index_optimal_policies :
                    plt.plot([0,1],values[:,index],label='policy '+str(index))
                plt.plot(self.specF['belief_vector'][:,0],alpha_max,'k:',label='optimal value')
                plt.legend()
                plt.xlabel("belief left")
                plt.ylabel(str(time_step+1)+" step policy tree value function")
                plt.savefig(self.specF['name_to_save_folder']+'/fig_'+str(time_step+1)+'step.png')
                plt.close()
                plt.figure()
                plt.plot(self.specF['belief_vector'][:,0],alpha_max,'k:',label='optimal value')
                plt.xlabel("belief left")
                plt.ylabel(str(time_step+1)+" step policy tree value function")
                plt.savefig(self.specF['name_to_save_folder']+'/fig_'+str(time_step+1)+'step_alphamax.png')
                plt.close()

            if terminate :
                numpy.savez_compressed(self.specF['name_to_save_folder']+'/Policies_over_time',Pruned_policies=Pruned_policies)
                print('THE SOLVER HAS CONVERGED')
                break

        if self.specF['discounting_parameter'] < 1 :
            self.plot_bellman_residual(List_residual)
        return

    def plot_bellman_residual(self,List_residual) :
        plt.figure()
        plt.plot(List_residual, label="Bellman residual")
        plt.plot(range(len(List_residual)),numpy.ones(len(List_residual))*self.specF['bellman_residual'],color='y',label='to be compared to')
        plt.xlabel("Time steps")
        plt.ylabel("Residual")
        plt.legend()
        plt.savefig(self.specF['name_to_save_folder']+'/fig_residual.png')
        plt.close()
        return

    def plot_plan_graph(self,initial_belief) : # to run only if value iteration has converged
        Pruned_policies = numpy.load(self.specF['name_to_save_folder']+'/Policies_over_time.npz',allow_pickle=True)['Pruned_policies']
        G=pgv.AGraph(strict=False,directed=True)
        for index in range(len(Pruned_policies)) :
            Pruned_policies[index].set_change_key()
        for policy_index in Pruned_policies :
            vector_map = policy_index.specF['optimal_alpha_map']
            policy_index.specF['optimal_alpha_map'] = self.simplify_optimal_alpha_map(vector_map)
            for successors_index in policy_index.specF['successors'] :
                vector_map = successors_index.specF['optimal_alpha_map']
                successors_index.specF['optimal_alpha_map'] = self.simplify_optimal_alpha_map(vector_map)
        name=""
        # CASE OF A GIVEN INITIAL BELIEF ---------------
        List_nodes_with_initial_belief = []
        if initial_belief :
            name = "_with_initial_belief"
            value_initial_belief = self.get_initial_belief_state()
            for index in range(len(Pruned_policies)) :
                if value_initial_belief[0]>Pruned_policies[index].specF['optimal_alpha_map'][0] and value_initial_belief[0]<Pruned_policies[index].specF['optimal_alpha_map'][1]:
                    initial_policy=index
                    List_nodes_with_initial_belief.append(initial_policy)
                    break
            Old_list = []
            while Old_list!=List_nodes_with_initial_belief :
                Old_list = List_nodes_with_initial_belief.copy()
                for index_in_the_list in List_nodes_with_initial_belief :
                    for index2 in range(len(Pruned_policies)) :
                        if numpy.allclose(Pruned_policies[index2].specF['optimal_alpha_map'],Pruned_policies[index_in_the_list].specF['successors'][0].specF['optimal_alpha_map'],atol=self.specF['precision_parameter']*10) or numpy.allclose(Pruned_policies[index2].specF['optimal_alpha_map'],Pruned_policies[index_in_the_list].specF['successors'][1].specF['optimal_alpha_map'],atol=self.specF['precision_parameter']*10):
                            if index2 not in List_nodes_with_initial_belief :
                                List_nodes_with_initial_belief.append(index2)
        # ------------------------------------
        for index in range(len(Pruned_policies)) :
            if initial_belief :
                if index in List_nodes_with_initial_belief :
                    G = self.build_nodes_graph(G,Pruned_policies,index)
            else :
                G = self.build_nodes_graph(G,Pruned_policies,index)
        G.draw(self.specF['name_to_save_folder']+'/plan_graph'+name+'.png',prog='dot')
        return

    def simplify_optimal_alpha_map(self,vector_map) :
        value_min = numpy.amin(vector_map)
        value_max = numpy.amax(vector_map)
        return numpy.asarray([value_min,value_max])

    def build_nodes_graph(self,G,Pruned_policies,index) :
        found_the_follower=False
        found_the_follower1 = False
        found_the_follower2 = False
        if Pruned_policies[index].specF['successors'][0].id==Pruned_policies[index].specF['successors'][1].id :
            print("same")
            for index2 in range(len(Pruned_policies)) :
                if numpy.allclose(Pruned_policies[index2].specF['optimal_alpha_map'],Pruned_policies[index].specF['successors'][0].specF['optimal_alpha_map'],atol=self.specF['precision_parameter']*10) :
                    print("equal1")
                    G.add_edge(Pruned_policies[index].id,Pruned_policies[index2].id)
                    n=G.get_node(Pruned_policies[index].id)
                    n.attr['label'] = Pruned_policies[index].key
                    nsucc0=G.get_node(Pruned_policies[index2].id)
                    nsucc0.attr['label'] = Pruned_policies[index2].key  
                    e=G.get_edge(Pruned_policies[index].id,Pruned_policies[index2].id)
                    e.attr['label']='TL/TR'
                    found_the_follower=True
                    break
        else :
            for index2 in range(len(Pruned_policies)) :
                if numpy.allclose(Pruned_policies[index2].specF['optimal_alpha_map'],Pruned_policies[index].specF['successors'][0].specF['optimal_alpha_map'],atol=self.specF['precision_parameter']*10) :
                    print("equal2")
                    G.add_edge(Pruned_policies[index].id,Pruned_policies[index2].id)
                    n=G.get_node(Pruned_policies[index].id)
                    n.attr['label'] = Pruned_policies[index].key
                    nsucc0=G.get_node(Pruned_policies[index2].id)
                    nsucc0.attr['label'] = Pruned_policies[index2].key  
                    e0=G.get_edge(Pruned_policies[index].id,Pruned_policies[index2].id)
                    e0.attr['label']=Pruned_policies[index].specF['list_of_observations_and_actions'][0][1]
                    found_the_follower1=True
                if numpy.allclose(Pruned_policies[index2].specF['optimal_alpha_map'],Pruned_policies[index].specF['successors'][1].specF['optimal_alpha_map'],atol=self.specF['precision_parameter']*10) :
                    print("equal3")
                    G.add_edge(Pruned_policies[index].id,Pruned_policies[index2].id)
                    n=G.get_node(Pruned_policies[index].id)
                    n.attr['label'] = Pruned_policies[index].key
                    nsucc1=G.get_node(Pruned_policies[index2].id)
                    nsucc1.attr['label'] = Pruned_policies[index2].key 
                    e1=G.get_edge(Pruned_policies[index].id,Pruned_policies[index2].id)
                    e1.attr['label']=Pruned_policies[index].specF['list_of_observations_and_actions'][1][1]
                    found_the_follower2=True
        if found_the_follower :
            print("found the follower")
        else :
            if found_the_follower1 and found_the_follower2 :
                print("found the followers 1 and 2")
            else :
                print("did not find followers")
                pdb.set_trace()
        return G



    def get_transition_matrix(self): # |A| x |S| x |S'| matrix,  action left, right, listen, state left, right
        return numpy.array([
            [[self.specF['proba_reset_tiger'] , self.specF['proba_reset_tiger']], [self.specF['proba_reset_tiger'] , self.specF['proba_reset_tiger']]],
            [[self.specF['proba_reset_tiger'] , self.specF['proba_reset_tiger']], [self.specF['proba_reset_tiger'] , self.specF['proba_reset_tiger']]],
            [[1.0, 0.0], [0.0, 1.0]]
        ])


    def get_observation_matrix(self): # |A| x |S| x |O| matrix, action left, right, listen, state left, right, observation left, right
        return numpy.array([
            [[self.specF['proba_of_correct_observation_after_opening_a_door'], self.specF['proba_of_correct_observation_after_opening_a_door']], [self.specF['proba_of_correct_observation_after_opening_a_door'], self.specF['proba_of_correct_observation_after_opening_a_door']]],
            [[self.specF['proba_of_correct_observation_after_opening_a_door'], self.specF['proba_of_correct_observation_after_opening_a_door']], [self.specF['proba_of_correct_observation_after_opening_a_door'], self.specF['proba_of_correct_observation_after_opening_a_door']]],
            [[self.specF['proba_of_correct_listening'], 1-self.specF['proba_of_correct_listening']], [1-self.specF['proba_of_correct_listening'], self.specF['proba_of_correct_listening']]]
        ])

    def get_reward_matrix(self): #|A| x |S| matrix, action left, right, listen, state left, right
        return numpy.array([[self.specF['reward_from_opening_the_wrong_door'],self.specF['reward_from_opening_the_correct_door']],[self.specF['reward_from_opening_the_correct_door'],self.specF['reward_from_opening_the_wrong_door']],[self.specF['reward_from_listening'], self.specF['reward_from_listening']]])


    def get_initial_belief_state(self):
        return numpy.array([0.5, 0.5])



    def update_Belief_Heuristic_Search(self,observation,belief_vector_left_right) : # Paragraph 3.3 of Kaelbling 1998 paper
        b_R_prior = belief_vector_left_right
        b_L_prior = 1-self.specF['belief_state']
        proba_observation_R = self.specF['proba_of_correct_listening']*belief_vector_left_right[0] + (1-self.specF['proba_of_correct_listening'])*belief_vector_left_right[1]
        proba_observation_L = self.specF['proba_of_correct_listening']*belief_vector_left_right[1] + (1-self.specF['proba_of_correct_listening'])*belief_vector_left_right[0]
        if observation == "TR" :
            b_R_posterior = self.specF['proba_of_correct_listening']*belief_vector_left_right[0]/proba_observation_R
        elif observation == "TL" :
            b_R_posterior = (1-self.specF['proba_of_correct_listening'])*belief_vector_left_right[0]/proba_observation_L # b_R_posterior+b_L_posterior=1
        else : pdb.set_trace()
        return numpy.asarray([1-b_R_posterior,b_R_posterior]), numpy.asarray([proba_observation_L,proba_observation_R])



    def compute_HeuristicSearch(): # Hansen 1998
        step_of_iteration = 0
        transition_matrix = self.get_transition_matrix()
        reward_matrix = self.get_reward_matrix()
        observation_matrix = self.get_observation_matrix()
        terminate = False
        List_residual = []

        # INITIALISATION : specify an initial FSC, the initial FSC I choose is going to be constituted of the action listen as a node, with connectivity on themselves
        # Difference with policy iteration, we compute everything only for point-evaluated belief states
        MyFSC = FiniteStateController({'possible_observations': self.specF['possible_observations'],'state_space':self.specF['state_space'],'heuristic':True})
        # Policy evaluation of the initial state |S| x |nodes|
        index_action=2
        new_action = 'LISTEN'
        successors=numpy.asarray([0,0])
        sum_over_observations = numpy.multiply(observation_matrix[index_action,:,0],reward_matrix[index_action,:]) + numpy.multiply(observation_matrix[index_action,:,1],reward_matrix[index_action,:])
        sum_over_transitions = numpy.dot(transition_matrix[index_action,:,:],sum_over_observations) # dim 2*1
        new_values = reward_matrix[index_action]+self.specF['discounting_parameter']*sum_over_transitions # dim 2*1
        new_node_id = MyFSC.create_new_node(new_action, successors, new_values,self.specF['initial_belief_state'])

        MyFSC.plot(self.specF['name_to_save_folder'],step_of_iteration) 
        MyFSC.update_value_of_states() 
        recompute_policy_eval = False
        # Computing value of node 0
        MyFSC.specF['list_of_nodes'][0].update_lower_bound()    
        MyFSC.specF['list_of_nodes'][0].specF['reach_proba']=1
        MyFSC.specF['list_of_nodes'][0].specF['depth']=0
        MyFSC.specF['list_of_nodes'][0].specF['node_belief'][0]!=0.5 : pdb.set_trace()
        self.specF['heuristic_search_error'] = MyFSC.specF['list_of_nodes'][0]['upper_bound']-MyFSC.specF['list_of_nodes'][0]['lower_bound']
        List_residual.append(self.specF['heuristic_search_error'])
        MyFSC.specF['fringes_of_fsc']=numpy.asarray([0])


        while terminate==False :

            print('\n--------- ITERATION STEP ---------'+str(step_of_policy_iteration))

            # POLICY EVALUATION of the FSC
            if recompute_policy_eval :
                value_has_converged = numpy.zeros(len(MyFSC.specF['list_of_nodes']),dtype=bool)
                while numpy.all(value_has_converged)==False: # while the elements of value_has_converged are not all equal to True
                    for nodes_in_fsc in MyFSC.specF['list_of_nodes'] :
                        index_node = MyFSC.specF['list_of_nodes'].index(nodes_in_fsc)
                        if value_has_converged[index_node]==False :
                            old_value = nodes_in_fsc.specF['value_of_states']
                            index_action = numpy.where(nodes_in_fsc.specF['action']==self.specF['action_space'])[0][0]
                            node1 = MyFSC.specF['list_of_nodes'][MyFSC.find_node_from_nodeid(nodes_in_fsc.specF['successors'][0])]
                            node2 = MyFSC.specF['list_of_nodes'][MyFSC.find_node_from_nodeid(nodes_in_fsc.specF['successors'][1])]
                            new_values = self.estimate_value_of_node(index_action,node1,node2,observation_matrix,transition_matrix,reward_matrix)
                            if numpy.sum(numpy.equal(new_values,old_value))!=new_values.shape[0] : # the value changed
                                value_has_converged[index_node] = False
                                nodes_in_fsc.specF['value_of_states'] = new_values
                            else :
                                value_has_converged[index_node] = True

                MyFSC.update_value_of_states()
                for nodes_in_fsc in MyFSC.specF['list_of_nodes'] :
                    nodes_in_fsc.update_lower_bound()


            # POLICY IMPROVEMENT
            lower_bound_previous = MyFSC.specF['list_of_nodes'][MyFSC.find_node_from_nodeid(self.specF['root_node_id'])].specF['lower_bound'].copy()
            # (a) FORWARD SEARCH
            number_of_forward_search = 0
            List_new_nodes_from_forward_search_id = []
            lower_bound_forward_search = lower_bound_previous.copy()

            # Compute Bellman Residual and upper bounds
            List_nodes_previous_delta = []
            List_upper_bounds = []
            for nodes_in_fsc in MyFSC.specF['list_of_nodes']:
                List_nodes_previous_delta.append(nodes_in_fsc)
                value_diff = nodes_in_fsc.specF['lower_bound']-nodes_in_fsc.specF['previous_lower_bound']
                List_upper_bounds.append(value_diff)
            Array_upper_bounds = numpy.absolute(numpy.asarray(List_upper_bounds))
            max_residual = numpy.amax(Array_upper_bounds)
            for nodes_in_fsc in enumerate(MyFSC.specF['list_of_nodes']) :
                value_upper_bound_for_this_node = nodes_in_fsc.specF['lower_bound']+max_residual*self.specF['discounting_parameter']/(1-self.specF['discounting_parameter'])
                nodes_in_fsc.update_upper_bound(value_upper_bound_for_this_node)


            while self.specF['heuristic_search_error'] > self.specF['residual_epsilon'] and lower_bound_forward_search >= lower_bound_previous:

                # decide which fringe node to expand
                Best_first_value = 0
                for nodes_in_fringes_id in MyFSC.specF['list_of_nodes']:
                    node_in_fringe = MyFSC.find_node_from_nodeid(nodes_in_fringes_id)
                    Expansion_value = (node_in_fringe.specF['upper_bound'] - node_in_fringe.specF['lower_bound'])*node_in_fringe.specF['reach_proba']*self.specF['discounting_parameter']**(node_in_fringe.specF['depth'])
                    if Expansion_value>Best_first_value :
                        expansion_node = nodes_in_fringes_id

                # expand it, and compute belief state, reach proba, depth, then back up value
                first=0
                for index_observation, new_observation in enumerate(self.specF['possible_observations']):
                    for node_suppl in MyFSC.specF['list_of_nodes'] :
                        obs2_id = node.id
                        if index_observation==0 : 
                            node1=MyFSC.find_node_from_nodeid(expansion_node)
                            node2=node_suppl
                        elif index_observation==1 :
                            node1 = node_suppl
                            node2 = MyFSC.find_node_from_nodeid(expansion_node)
                        for index_action, new_action in enumerate(self.specF['action_space']) :
                            new_value_if_that_node_was_root = self.estimate_value_of_node(index_action,node1,node2,observation_matrix,transition_matrix,reward_matrix)
                            value_of_belief_node = numpy.multiply(self.specF['initial_belief_state'],new_value_if_that_node_was_root)
                            if first==0 :
                                values_of_states = new_value_if_that_node_was_root.copy()
                                value_of_new_node = value_of_belief_node.copy()
                                List_action_obs1_ob2 = [new_action, node1.id, node2.id] 
                                first=1
                            elif value_of_belief_node > value_of_new_node :
                                values_of_states = new_value_if_that_node_was_root.copy()
                                value_of_new_node = value_of_belief_node.copy()
                                List_action_obs1_ob2 = [new_action, node1.id, node2.id] 

                self.specF['root_node_id'] = MyFSC.create_new_node(List_action_obs1_ob2[0], numpy.asarray([List_action_obs1_ob2[1], List_action_obs1_ob2[2]]), values_of_states,self.specF['initial_belief_state'])
                List_new_nodes_from_forward_search_id.append(self.specF['root_node_id'].copy())

                # back up belief state values
                List_of_backup_nodes_id = []
                List_node_to_back_up = [self.specF['root_node_id']]
                depth_from_the_root = 1

                while len(List_node_to_back_up)>0:
                    node_to_back_up = List_node_to_back_up[0]
                    List_node_to_back_up.remove(node_to_back_up)
                    successors = MyFSC.specF['list_of_nodes'][MyFSC.find_node_from_nodeid(node_to_back_up)].specF['successors']
                    for index_observation, new_observation in enumerate(self.specF['possible_observations']) :
                        if successors[index_observation] not in List_of_backup_nodes_id :
                            node_successor = MyFSC.specF['list_of_nodes'][MyFSC.find_node_from_nodeid(successors[index_observation])]
                            # update the belief of the successor, as well as its reach proba and depth
                            posterior_belief, proba_observations = self.update_Belief_Heuristic_Search(new_observation,MyFSC.specF['list_of_nodes'][MyFSC.find_node_from_nodeid(node_to_back_up)].specF['node_belief']) 
                            print('TEST PROBA OBS')
                            print(proba_observations)
                            pdb.set_trace()
                            MyFSC.specF['list_of_nodes'][MyFSC.find_node_from_nodeid(successors[index_observation])].specF['node_belief'] = posterior_belief
                            MyFSC.specF['list_of_nodes'][MyFSC.find_node_from_nodeid(successors[index_observation])].specF['depth'] = depth_from_the_root
                            MyFSC.specF['list_of_nodes'][MyFSC.find_node_from_nodeid(successors[index_observation])].specF['reach_proba'] = MyFSC.specF['list_of_nodes'][MyFSC.find_node_from_nodeid(node_to_back_up)].specF['reach_proba']*proba_observations[index_observation]
                            # add the updated node in the list
                            List_of_backup_nodes_id.append(successors[index_observation])
                            if successors[index_observation] not in List_node_to_back_up :
                                List_node_to_back_up.append(successors[index_observation])
                    depth_from_the_root+=1

                # back up lower and upper bounds from the leaves of the search tree, to the root
                # Compute Bellman Residual and upper bounds
                List_upper_bounds = []
                for nodes_in_fsc in MyFSC.specF['list_of_nodes']:
                    nodes_in_fsc.update_lower_bound()
                    value_diff = nodes_in_fsc.specF['lower_bound']-nodes_in_fsc.specF['previous_lower_bound']
                    List_upper_bounds.append(value_diff)
                Array_upper_bounds = numpy.absolute(numpy.asarray(List_upper_bounds))
                max_residual = numpy.amax(Array_upper_bounds)
                for nodes_in_fsc in enumerate(MyFSC.specF['list_of_nodes']) :
                    value_upper_bound_for_this_node = nodes_in_fsc.specF['lower_bound']+max_residual*self.specF['discounting_parameter']/(1-self.specF['discounting_parameter'])
                    nodes_in_fsc.update_upper_bound(value_upper_bound_for_this_node)

                # testing termination of while loop
                lower_bound_forward_search = MyFSC.specF['list_of_nodes'][MyFSC.find_node_from_nodeid(self.specF['root_node_id'])].specF['lower_bound'].copy()
                self.specF['heuristic_search_error'] = MyFSC.self.specF['list_of_nodes'][MyFSC.find_node_from_nodeid(self.specF['root_node_id'])]['upper_bound']-MyFSC.self.specF['list_of_nodes'][MyFSC.find_node_from_nodeid(self.specF['root_node_id'])]['lower_bound']

            # (b) TEST TERMINATION
            # The error bound is the difference between the upper and lower bounds on the value of the starting belief state
            List_residual.append(self.specF['heuristic_search_error'])
            if self.specF['heuristic_search_error'] < self.specF['residual_epsilon'] :
                terminate = True
                print("HEURISTIC SEARCH TERMINATED AT STEP "+str(step_of_iteration))
                break # exit the while

            # (c) Cleaning the reachable nodes, from the leaves to the root, old nodes are List_previousdelta_nodes_id, new added nodes are List_new_nodes_from_forward_search_id 
            recompute_policy_eval = True
            max_depth = MyFSC.compute_max_depth()
            while max_depth>=0 :
                for reachable_node_id in List_new_nodes_from_forward_search_id :
                    reachable_node = MyFSC.find_node_from_nodeid(reachable_node_id)
                    depth_for_this_node = reachable_node.specF['depth']
                    if depth_for_this_node==max_depth :
                        self.clean_fsc_from_new_nodes(MyFSC, List_nodes_previous_delta, reachable_node)
                max_depth-=1


            #(d) PRUNING 
            List_of_new_nodeid_optimizing_starting_belief_state = [self.specF['root_node_id']]
            List_node_ongoing = [self.specF['root_node_id']]
            for node in List_node_ongoing :
                List_node_ongoing.delete(node)
                successors = MyFSC.specF['list_of_nodes'][MyFSC.find_node_from_nodeid(node)].specF['successors']
                for index_observation, new_observation in enumerate(self.specF['possible_observations']) :
                    if successors[index_observation] not in List_of_new_nodeid_optimizing_starting_belief_state :
                        List_of_new_nodeid_optimizing_starting_belief_state.append(successors[index_observation])
                        List_node_ongoing.append(successors[index_observation])

            for node in MyFSC.specF['list_of_nodes'] :
                if node.id in List_of_new_nodeid_optimizing_starting_belief_state :
                    continue
                else :
                    MyFSC.remove_node(node)

            MyFSC.update_value_of_states()
            MyFSC.plot(self.specF['name_to_save_folder'],step_of_iteration)   

        self.plot_bellman_residual(List_residual)

        return


    def test_pointwise_dominance_heuristic(self, Previous_fsc_delta, new_node_fsc) : 
        list_of_domination = []
        new_value = new_node_fsc.specF['value_of_states']
        for node in Previous_fsc_delta :
            if numpy.sum(numpy.less_equal(node.specF['value_of_states']+numpy.ones(node.specF['value_of_states'].shape[0])*self.specF['precision_pointwise_dominance'],new_value))== new_value.shape[0] :
                list_of_domination.append(node.id)
        return list_of_domination


    def clean_fsc_from_new_nodes(self, FSC, Previous_fsc_delta, new_node_fsc):
        # Test if the action and the successor links associated with it are the same as those of a machine state already in FSC (Delta in Hansen)
        test1, nodeid = self.test_equal_machine_state_heuristic(new_node_fsc,Previous_fsc_delta)
        if test1 :
            for node in FSC.specF['list_of_nodes'] :
                for index_observation, observation in self.specF['possible_observations'] :
                    if node.specF['successors'][index_observation]==new_node_fsc.id :
                        node.specF['successors'][index_observation] = nodeid
            FSC.remove_node(new_node_fsc)
        else : # Else test if the vector pointwise dominates a vector associated with a machine state in MyFSC
            liste_of_domination_test2 = self.test_pointwise_dominance_heuristic(Previous_fsc_delta, new_node_fsc)
            if len(liste_of_domination_test2)!=0 : # a new vector dominates old vectors
                if len(liste_of_domination_test2)==1 : # only one old vector is dominated, the old vector will be changed
                    nodeid = liste_of_domination_test2[0]
                    node_to_change = FSC.specF['list_of_nodes'][FSC.find_node_from_nodeid(nodeid)]
                    node_to_change.modify_node_with_belief(new_node_fsc.specF['action'],new_node_fsc.specF['successors'],new_node_fsc.specF['value_of_states'], new_node_fsc.specF['node_belief'])
                    for node in FSC.specF['list_of_nodes'] :
                        for index_observation, observation in self.specF['possible_observations'] :
                            if node.specF['successors'][index_observation]==new_node_fsc.id :
                                node.specF['successors'][index_observation] = nodeid
                else : # more than one old vector is dominated, one old vector will be changed, and the other one deleted
                    selected_node=False
                    List_of_node_to_merge = []
                    for other_node_id in liste_of_domination_test2 :
                        flattened_ids = numpy.asarray(List_of_node_to_merge.copy())
                        flattened_ids = flattened_ids.flatten()
                        if (other_node_id not in flattened_ids) and selected_node==True:
                            List_of_node_to_merge.append(other_node_id)
                        if (other_node_id not in flattened_ids) and selected_node==False:
                            nodeid_to_merge_in = other_node_id
                            node_to_change = FSC.specF['list_of_nodes'][FSC.find_node_from_nodeid(nodeid_to_merge_in)]
                            node_to_change.modify_node_with_belief(new_node_fsc.specF['action'],new_node_fsc.specF['successors'],new_node_fsc.specF['value_of_states'], new_node_fsc.specF['node_belief'])
                            for node in FSC.specF['list_of_nodes'] :
                                for index_observation, observation in self.specF['possible_observations'] :
                                    if node.specF['successors'][index_observation]==new_node_fsc.id :
                                        node.specF['successors'][index_observation] = nodeid_to_merge_in
                            selected_node=True
                    if selected_node==False :
                        print("PROBLEM SELECTED NODE")
                        pdb.set_trace()

                    # Merging the dominated machine states
                    if len(List_of_node_to_merge)!=0 :
                        for node_merging_id in List_of_node_to_merge :
                            for node in FSC.specF['list_of_nodes'] :
                                for index_observation, observation in self.specF['possible_observations'] :
                                    if node.specF['successors'][index_observation]==node_merging_id :
                                        node.specF['successors'][index_observation] = nodeid_to_merge_in

                            node_to_delete = FSC.find_node_from_nodeid(node_merging_id)
                            FSC.remove_node(FSC.specF['list_of_nodes'][node_to_delete])

                FSC.remove_node(new_node_fsc)
            else : # no domination at all, we keep this new machine state
                print('machine state '+str(new_node_fsc.id)+' kept into the FSC')            

        # recompute depth for all machine states



 






