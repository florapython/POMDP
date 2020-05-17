# Flora Bouchacourt

import numpy, os, pdb, pylab
from Node import *
import pygraphviz as pgv


class FiniteStateController:
	""" Finite State Controller """

	def __init__(self, spec) : 
		self.specF={} #empty dic
		self.specF['counter'] = spec.get('counter',0)
		self.specF['list_of_nodes'] = spec.get('list_of_nodes',[])
		self.specF['value_of_states'] =  spec.get('value_of_states',None) # 2 * number of nodes
		#self.specF['previous_value_of_states'] = spec.get('previous_value_of_states',None)
		self.specF['array_belief_states'] = spec.get('array_belief_states',None) # in case we are in heuristic search
		self.specF['belief_vector'] = spec.get('belief_vector',None) # if None, we are doing heuristic search
		self.specF['possible_observations'] = spec.get('possible_observations',None)
		self.specF['state_space'] = spec.get('state_space',None)
		self.specF['heuristic'] = spec.get('heuristic',False) # in case we are going to use Heuristic search to expand the FSC
		self.specF['fringes_of_fsc'] = spec.get('fringes_of_fsc',None)
		self.specF['root_node_id'] = 0
		self.specF['max_iteration_step'] = spec.get('max_iteration_step',None)
		self.specF['precision_param'] = 0.001
		self.update_value_of_states()


	"""
	def create_new_node(self,action,successors,values_of_states,node_belief) : # action is a string, like 'LISTEN'
		MyNewNode = Node({'action':action,'value_of_states':values_of_states,'successors':successors,'node_belief':node_belief},self.specF['counter'])
		self.specF['list_of_nodes'].append(MyNewNode)
		#self.update_value_of_states()
		self.specF['counter'] +=1
		return MyNewNode.id
	"""

	
	def compute_max_diff_value_of_states(self,List_nodes_id_to_consider) :
		possible_max_residuals = []
		belief_array = numpy.asarray([numpy.arange(0,1.0001,0.01),1-numpy.arange(0,1.0001,0.01)]).T
		for index_node, node in enumerate(self.specF['list_of_nodes']) :
			if node.id in List_nodes_id_to_consider :
				print(node.id)
				print(node.specF['value_of_states'])
				print(node.specF['previous_value_of_states'])
				print("DEBUT COMPUTE MAX")
				#pdb.set_trace()
				diff_value = node.specF['value_of_states']-node.specF['previous_value_of_states']
				max_now = numpy.amax(numpy.abs(diff_value))
				possible_max_residuals.append(max_now)
				# Check that this max is larger than the max over belief state
				diff_value_over_all_belief_states = numpy.dot(belief_array,diff_value)
				max_for_all_beliefs = numpy.amax(numpy.abs(diff_value_over_all_belief_states))
				if numpy.greater_equal(max_for_all_beliefs, max_now+self.specF['precision_param']) : 
					print("\n\n\n\n\n\n\n ---------PROBLEME COMPUTE MAX DIFF IN FSC CLASS---------\n\n\n\n\n\n\n")
					pdb.set_trace()
		self.specF['max_iteration_step'] = numpy.amax(numpy.asarray(possible_max_residuals))
		print(self.specF['max_iteration_step'])
		print("MAX ITERATION STEP")
		#pdb.set_trace()
		return
	

	def compute_max_diff_value_of_states_all(self) :
		possible_max_residuals = []
		belief_array = numpy.asarray([numpy.arange(0,1.0001,0.01),1-numpy.arange(0,1.0001,0.01)]).T
		for index_node, node in enumerate(self.specF['list_of_nodes']) :
			diff_value = node.specF['value_of_states']-node.specF['previous_value_of_states']
			max_now = numpy.amax(numpy.abs(diff_value))
			possible_max_residuals.append(max_now)
			# Check that this max is larger than the max over belief state
			diff_value_over_all_belief_states = numpy.dot(belief_array,diff_value)
			max_for_all_beliefs = numpy.amax(numpy.abs(diff_value_over_all_belief_states))
			print(numpy.greater_equal(max_for_all_beliefs, max_now+self.specF['precision_param']))
			if numpy.greater_equal(max_for_all_beliefs, max_now+self.specF['precision_param']) :
				print("\n\n\n\n\n\n\n ---------PROBLEME COMPUTE MAX DIFF IN FSC CLASS---------\n\n\n\n\n\n\n")
				pdb.set_trace()
		self.specF['max_iteration_step'] = numpy.amax(numpy.asarray(possible_max_residuals))
		print(self.specF['max_iteration_step'])
		print("MAX ITERATION STEP")
		#pdb.set_trace()
		return


	def create_new_node_with_bounds(self,action,successors,values_of_states,node_belief,values_of_states_upper_bound) : # action is a string, like 'LISTEN'
		MyNewNode = Node({'action':action,'value_of_states':values_of_states,'successors':successors,'node_belief':node_belief,'value_of_states_upper_bound':values_of_states_upper_bound},self.specF['counter'])
		self.specF['list_of_nodes'].append(MyNewNode)
		self.specF['counter'] +=1
		return MyNewNode.id


	def update_value_of_states(self) :
		if len(self.specF['list_of_nodes'])==0 :
			self.specF['value_of_states'] = None
			self.specF['array_belief_states'] = None
		else :
			Matrix = numpy.ones((self.specF['state_space'].shape[0],len(self.specF['list_of_nodes']))) # S x number of nodes
			for index_node, node in enumerate(self.specF['list_of_nodes']) :
				Matrix[:,index_node] = node.specF['value_of_states']
				node.key = node.write_key()
			self.specF['value_of_states'] = Matrix.copy()

		if self.specF['heuristic'] : # an array describing the belief of each node, during heuristic search, like update_value_of_states
			Matrix = numpy.ones((self.specF['state_space'].shape[0],len(self.specF['list_of_nodes']))) # S x number of nodes
			for index_node, node in enumerate(self.specF['list_of_nodes']) :
				Matrix[:,index_node] = node.specF['node_belief']
			self.specF['array_belief_states'] = Matrix.copy()
		return 



	def remove_node(self,node) :
		#index_of_node = self.specF['list_of_nodes'].index(node)
		self.specF['list_of_nodes'].remove(node)
		#self.update_value_of_states()
		return


	def change_action_and_successors(self,nodeid, action, successors, value) :
		location_node = self.find_node_from_nodeid(nodeid)
		node_to_change = self.specF['list_of_nodes'][location_node]
		node_to_change.modify_node(action, successors, value)
		#self.update_value_of_states()
		return

	def find_node_from_nodeid(self,nodeid) : # give the index of the node when given the node id
		location_node = 0
		found = False
		for node_to_test in self.specF['list_of_nodes'] :
			if node_to_test.id == nodeid :
				found = True
				return location_node
			location_node+=1
		if found==False : 
			print("PROBLEM IN FINDING NODE FROM NODEID "+str(nodeid))
			pdb.set_trace()
		return

	def compute_max_depth(self) :
		depth=0
		for node in self.specF['list_of_nodes'] :
			if node.specF['depth']>depth :
				depth = node.specF['depth']
		return depth


	def list_of_ids(self) :
		List_of_ids = []
		for node in self.specF['list_of_nodes'] :
			List_of_ids.append(node.id)
		return List_of_ids


	def list_of_belief_states(self) :
		List = []
		for node in self.specF['list_of_nodes'] :
			List.append(node.specF['node_belief'])
		return List

	def list_of_lower_values(self) :
		List = []
		for node in self.specF['list_of_nodes'] :
			List.append(node.specF['lower_bound'])
		return List


	def list_of_upper_values(self) :
		List = []
		for node in self.specF['list_of_nodes'] :
			List.append(node.specF['upper_bound'])
		return List

	def print_state_of_fsc(self) :
		#print("PRINT STATE OF FSC")
		print("There are "+str(len(self.specF['list_of_nodes']))+" nodes")
		print("Ids of nodes are ")
		print(self.list_of_ids())
		print("The current root node is "+str(self.specF['root_node_id']))
		print("List of lower values")
		print(self.list_of_lower_values())
		print("List of upper values")
		print(self.list_of_upper_values())
		#pdb.set_trace()
		return


	def plot(self,path,iteration_step) :
		G=pgv.AGraph(strict=False,directed=True)
		for node in self.specF['list_of_nodes'] :
			G.add_node(node.id)
			n=G.get_node(node.id)
			n.attr['label'] = node.key
			if node.specF['successors'][0] == node.specF['successors'][1]:
				successor_node_location = self.find_node_from_nodeid(node.specF['successors'][0])
				successor_node = self.specF['list_of_nodes'][successor_node_location]
				G.add_edge(node.id,successor_node.id)
				nsucc0=G.get_node(successor_node.id)
				nsucc0.attr['label'] = successor_node.key
				e=G.get_edge(node.id,successor_node.id)
				e.attr['label']=self.specF['possible_observations'][0]+'/'+self.specF['possible_observations'][1]
			else :
				successor_node_location1 = self.find_node_from_nodeid(node.specF['successors'][0])
				successor_node1 = self.specF['list_of_nodes'][successor_node_location1]
				successor_node_location2 = self.find_node_from_nodeid(node.specF['successors'][1])
				successor_node2 = self.specF['list_of_nodes'][successor_node_location2]
				G.add_edge(node.id,successor_node1.id)
				G.add_edge(node.id,successor_node2.id)
				nsucc1=G.get_node(successor_node1.id)
				nsucc1.attr['label'] = successor_node1.key
				e1=G.get_edge(node.id,successor_node1.id)
				e1.attr['label']=self.specF['possible_observations'][0]
				nsucc2=G.get_node(successor_node2.id)
				nsucc2.attr['label'] = successor_node2.key
				e2=G.get_edge(node.id,successor_node2.id)
				e2.attr['label']=self.specF['possible_observations'][1]
		G.draw(path+'/FSC_iterationstep_'+str(iteration_step)+'.png',prog='dot')
		return




