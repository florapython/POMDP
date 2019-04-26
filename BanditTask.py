# Flora Bouchacourt

# Class that runs a bandit task

import numpy, os

class BanditTask:
    """ Class that runs a bandit task episode. An episode is composed of trials where the p values of the arms are constant for now.  """
    def __init__(self,spec):
    self.specF={} #empty dic
    self.specF['name_to_save_folder']=spec.get('name_to_save_folder','test')
    if not os.path.exists(self.specF['name_to_save_folder']+'/'):
        os.makedirs(self.specF['name_to_save_folder']+'/')
    self.specF['number_of_arms'] = spec.get('number_of_arms',2)
    self.specF['distribution_for_Pvalues'] = spec.get('distribution_for_ps','uniform')
    self.specF['bernoulli'] = self.get('bernoulli',True)
    self.specF['gaussian_bandit'] = self.get('gaussian_bandit',False)
    self.specF['variance_for_pooling_arms'] = spec.get('variance_for_pooling_arms',1)  # in case arms are not bernouilli but rewards are drawn from a gaussian distribution around a mean value
    self.specF['termination'] = spec.get('termination',"constant_proba")
    self.specF['proba_of_termination'] = spec.get('proba_of_termination',0.05)
    self.specF['trial_number_for_termination'] = spec.get('trial_number_for_termination',0)

    self.PvaluesOfArms = self.draw_ArmProbaDistribForReward()


    def draw_ArmProbaDistribForReward(self) : # choose the p1,p2 values of each arm in the bandit task from uniform distribution
        if self.specF['distribution_for_Pvalues']=='uniform' :
            return numpy.random.uniform(0,1,size=self.specF['number_of_arms'])
        if self.specF['distribution_for_Pvalues']=='gaussian' : # need to write this script later
            pass

    def pull_AnArm(self,arm_number) : 
    # arm_number is the integer, 0 or 1, or the arm pulled at t
    # PvaluesOfArms is the array of p1,p2 describing the bandit
        if self.specF['bernoulli'] :
            random_number = numpy.random.uniform(0,1)
            if random_number<=self.PvaluesOfArms[arm_number] :
                return 1
            else :
                return 0

    def test_Termination(self,time_step) :
        termination = False
        # test if the episode terminates
        if self.specF['termination']=="constant_proba" :
            random_number = numpy.random.uniform(0,1)
            if random_number <= self.specF['proba_of_termination'] :
                termination=True
        if self.specF['termination']=="constant_trial_number" :
            if time_step==self.specF['trial_number_for_termination'] :
                termination=True
        else : 
            raise ValueError('Error in BanditTask.test_Termination : Termination criteria not defined')
        return termination





