# Flora Bouchacourt

import numpy, os, pdb, pylab
from matplotlib import pyplot as plt
from PolicyTree import *
import networkx as nx
import graphviz
from networkx.drawing import *
from networkx.drawing.nx_agraph import graphviz_layout
import pygraphviz as pgv

class Tiger:
    """ Class that runs a Tiger task episode.  """
    def __init__(self,spec):
        self.specF={} #empty dic
        self.specF['name_to_save_folder']=spec.get('name_to_save_folder','test')
        if not os.path.exists(self.specF['name_to_save_folder']+'/'):
            os.makedirs(self.specF['name_to_save_folder']+'/')
        self.specF['number_of_doors'] = spec.get('number_of_doors',2)
        if self.specF['number_of_doors'] != 2 :
            print("\nERROR : The class has to be rewritten for more than 2 doors")
            pdb.set_trace()
        self.specF['number_of_actions'] = spec.get('number_of_actions',3)
        self.specF['action_space'] = spec.get('action_space',numpy.asarray(['LEFT','RIGHT','LISTEN']))
        self.specF['state_space'] = spec.get('state_space',numpy.asarray(['LEFT','RIGHT']))
        self.specF['possible_observations'] = spec.get('possible_observations',numpy.asarray(['TL','TR']))
        self.specF['proba_of_correct_listening'] = spec.get('proba_of_correct_listening',0.85)
        self.specF['proba_of_correct_observation_after_opening_a_door']  = spec.get('proba_of_correct_observation_after_opening_a_door',0.5)
        self.specF['reward_from_listening'] = spec.get('reward_from_listening',-1)
        self.specF['reward_from_opening_the_correct_door'] = spec.get('reward_from_opening_the_correct_door',10)
        self.specF['reward_from_opening_the_wrong_door'] = spec.get('reward_from_opening_the_wrong_door',-100)
        self.specF['time_step_in_episode'] = spec.get('time_step_in_episode',0)
        self.specF['belief_state'] = spec.get('belief_state',0.5) # belief that the tiger is behind the right door, the belief for state left is 1-belief
        self.specF['proba_reset_tiger'] = spec.get('proba_reset_tiger',0.5)
        self.specF['discounting_parameter'] = spec.get('discounting_parameter',1.0)
        if self.specF['discounting_parameter']>1 or self.specF['discounting_parameter']<0 :
            print("\nERROR : The discounting parameter takes value between 0 and 1")
            pdb.set_trace()
        self.specF['belief_vector'] = spec.get('belief_vector',numpy.vstack((numpy.arange(0,1.005,0.01),1-numpy.arange(0,1.005,0.01))))
        self.specF['tiger_location'] = self.decide_NewTigerLocation()
        self.specF['residual_epsilon'] = spec.get('residual_epsilon',0.1)
        self.specF['bellman_residual'] = self.specF['residual_epsilon']*(1-self.specF['discounting_parameter'])/self.specF['discounting_parameter']
        self.specF['give_specific_initial_belief'] = spec.get('give_specific_initial_belief',False)
    

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



    def compute_OptimalValue(self,Matrix, belief_space, List_of_policies) :
        MaxValue = numpy.ones(belief_space.shape[0])*numpy.nan
        List_index_optimal_policies = []
        optimal_alpha_map = {}
        for index, belief_value in enumerate(belief_space) :
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
                            if List_of_policies[index_policy1].specF['top_node_action']=='LISTEN' :
                                List_to_prune.append(index_policy2)
                            elif List_of_policies[index_policy2].specF['top_node_action']=='LISTEN' :
                                List_to_prune.append(index_policy1)
                            else :
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


    def compute_PolicyIteration(self,horizon) :

        return


    def compute_ValueIteration(self,horizon) :
        counter=0
        terminate = False
        transition_matrix = self.get_transition_matrix()
        reward_matrix = self.get_reward_matrix()
        observation_matrix = self.get_observation_matrix()
        G=pgv.AGraph(strict=False,directed=True)
        List_residual = []
        for time_step in range(horizon) :
            List_of_policies = []
            if time_step==0:
                """
                values_list=[]
                actions_list = []
                if self.specF['give_specific_initial_belief'] :
                    initial_belief = self.get_initial_belief_state()
                    value_of_belief =  initial_belief[0]*reward_matrix[:,0]+initial_belief[1]*reward_matrix[:,1]
                    if value_of_belief.shape[0]!=self.specF['action_space'].shape[0] :
                        pdb.set_trace()
                    initial_action = numpy.argmax(value_of_belief)
                    MyNewPolicy = PolicyTree({'top_node_action':self.specF['action_space'][initial_action],'list_of_observations_and_actions': [self.specF['action_space'][initial_action]], 'value_of_the_policy_tree':reward_matrix[initial_action,:]})
                    List_of_policies.append(MyNewPolicy)
                else :
                """
                for index, new_action in enumerate(self.specF['action_space']) :
                    MyNewPolicy = PolicyTree({'top_node_action':new_action,'list_of_observations_and_actions': [new_action], 'value_of_the_policy_tree':reward_matrix[index], 'horizon':time_step},counter)
                    counter+=1
                    List_of_policies.append(MyNewPolicy)
            else :
                for index_action, new_action in enumerate(self.specF['action_space']) : # action at the top of the new policy tree
                    for index_subpolicy1, subpolicy1 in enumerate(List_of_old_policies) : # subpolicy for observation left
                        for index_subpolicy2, subpolicy2 in enumerate(List_of_old_policies) : # subpolicy for observation right
                            List_of_current_subpolicies = [subpolicy1,subpolicy2]
                            # Creating the new list of observations and actions
                            NewList = []
                            for index_observation in range(self.specF['possible_observations'].shape[0]) :
                                old_policy = List_of_current_subpolicies[index_observation].specF['list_of_observations_and_actions'].copy()
                                old_policy.insert(0,self.specF['possible_observations'][index_observation])
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

            value_of_belief = numpy.dot(numpy.transpose(self.specF['belief_vector'].copy()),values)



            alpha_max,List_index_optimal_policies = self.compute_OptimalValue(value_of_belief,self.specF['belief_vector'][0],List_of_policies)
            print(str(len(List_index_optimal_policies))+" OPTIMAL POLICIES")


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
            if time_step < 5 or time_step > 100 :
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
            
            if time_step < 5 or time_step > 100 :
                plt.figure()
                for index in List_index_optimal_policies :
                    plt.plot(self.specF['belief_vector'][0],value_of_belief[:,index],label='policy '+str(index))
                plt.plot(self.specF['belief_vector'][0],alpha_max,'k:',label='optimal value')
                plt.legend()
                plt.xlabel("belief left")
                plt.ylabel(str(time_step+1)+" step policy tree value function")
                plt.savefig(self.specF['name_to_save_folder']+'/fig_'+str(time_step+1)+'step_newmethod.png')
                plt.close()
                plt.figure()
                plt.plot(self.specF['belief_vector'][0],alpha_max,'k:',label='optimal value')
                plt.xlabel("belief left")
                plt.ylabel(str(time_step+1)+" step policy tree value function")
                plt.savefig(self.specF['name_to_save_folder']+'/fig_'+str(time_step+1)+'step_alphamax.png')
                plt.close()

            if terminate :
                numpy.savez_compressed(self.specF['name_to_save_folder']+'/Policies_over_time',Pruned_policies=Pruned_policies)
                print('THE SOLVER HAS CONVERGED')
                break
        plt.figure()
        plt.plot(List_residual, label="Bellman residual")
        plt.plot(range(horizon),numpy.ones(horizon)*self.specF['bellman_residual'],color='y',label='to be compared to')
        plt.xlabel("Time steps")
        plt.ylabel("Residual")
        plt.legend()
        plt.savefig(self.specF['name_to_save_folder']+'/fig_residual.png')

        return


    def plot_plan_graph(self) : # to run only if value iteration has converged
        Pruned_policies = numpy.load(self.specF['name_to_save_folder']+'/Policies_over_time.npz')['Pruned_policies']
        G=pgv.AGraph(strict=False,directed=True)
        for index in range(len(Pruned_policies)) :
            Pruned_policies[index].set_change_key()
        for index in range(len(Pruned_policies)) :
            if Pruned_policies[index].specF['successors'][0].id==Pruned_policies[index].specF['successors'][1].id :
                for index2 in range(len(Pruned_policies)) :
                    if Pruned_policies[index2].specF['optimal_alpha_map']==Pruned_policies[index].specF['successors'][0].specF['optimal_alpha_map'] :
                        G.add_edge(Pruned_policies[index].id,Pruned_policies[index2].id)
                        n=G.get_node(Pruned_policies[index].id)
                        n.attr['label'] = Pruned_policies[index].key
                        nsucc0=G.get_node(Pruned_policies[index2].id)
                        nsucc0.attr['label'] = Pruned_policies[index2].key  
                        e=G.get_edge(Pruned_policies[index].id,Pruned_policies[index2].id)
                        e.attr['label']='TL/TR'
                        break
            else :
                for index2 in range(len(Pruned_policies)) :
                    if Pruned_policies[index2].specF['optimal_alpha_map']==Pruned_policies[index].specF['successors'][0].specF['optimal_alpha_map'] :
                        G.add_edge(Pruned_policies[index].id,Pruned_policies[index2].id)
                        n=G.get_node(Pruned_policies[index].id)
                        n.attr['label'] = Pruned_policies[index].key
                        nsucc0=G.get_node(Pruned_policies[index2].id)
                        nsucc0.attr['label'] = Pruned_policies[index2].key  
                        e0=G.get_edge(Pruned_policies[index].id,Pruned_policies[index2].id)
                        e0.attr['label']=Pruned_policies[index].specF['list_of_observations_and_actions'][0][1]
                    if Pruned_policies[index2].specF['optimal_alpha_map']==Pruned_policies[index].specF['successors'][1].specF['optimal_alpha_map'] :
                        G.add_edge(Pruned_policies[index].id,Pruned_policies[index2].id)
                        n=G.get_node(Pruned_policies[index].id)
                        n.attr['label'] = Pruned_policies[index].key
                        nsucc1=G.get_node(Pruned_policies[index2].id)
                        nsucc1.attr['label'] = Pruned_policies[index2].key 
                        e1=G.get_edge(Pruned_policies[index].id,Pruned_policies[index2].id)
                        e1.attr['label']=Pruned_policies[index].specF['list_of_observations_and_actions'][1][1]
        G.draw(self.specF['name_to_save_folder']+'/plan_graph.png',prog='dot')
        return

    def plot_plan_graph_with_initial_belief(self) : # to run only if value iteration has converged
        return



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








