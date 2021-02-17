"""
Created on May 17, 2016

@author: xiul, t-zalipt
"""

import json
import ast
from deep_dialog import dialog_config
from deep_dialog.agents import DQNAgent
from deep_dialog.dialog_system import StateTracker
from deep_dialog.nlg import nlg
from deep_dialog.nlu.nlu_ensemble_predict import *


class ComponentController:
    """ A dialog manager to mediate the interaction between an agent and a customer """
    
    def __init__(self, act_set, slot_set, movie_dictionary, params):
        self.agent = DQNAgent(movie_dictionary, act_set, slot_set, params, phase="inference")
        self.act_set = act_set
        self.slot_set = slot_set
        self.state_tracker = StateTracker(act_set, slot_set, movie_dictionary)
        self.user_action = None
        self.reward = 0
        self.episode_over = False
        self.params = params
        self.nlu = NluEnsemblePredict()
        self.turn = 0
        self.nlu_output = None
        self.fallback = False
        self.print_nlu = False

    def adapter(self, nlu_output):
        user_action = {}
        intent = nlu_output['intent']['name']
        if intent == 'nlu_fallback':
            return 'nlu_fallback'
        if intent in self.act_set:
            user_action['diaact'] = intent
            user_action['request_slots'] = {}
        else:
            user_action['diaact'] = 'request'
            entity_list = intent.split('+')
            entity_set = {}
            for entity in entity_list:
                entity_set[entity.strip()] = 'UNK'
            user_action['request_slots'] = entity_set 

        user_action['inform_slots'] = self.get_entity_set(nlu_output['entities'], nlu_output['text'])
        user_action['turn'] = self.turn
        for k in user_action['inform_slots'].keys():
                old = user_action['inform_slots'][k]
                user_action['inform_slots'][k] = old.lower()
                
        return user_action


    def get_entity_set(self, entity_list, text):
        possible_entity = ["starttime","actor","city","video_format","critic_rating","distanceconstraints","other","numberofkids","state",
                      "zip","theater_chain","mpaa_rating","theater","description","moviename","numberofpeople","date","genre","closing"]
        entity_set = {}
        for entity in entity_list:
            if entity['entity'] in possible_entity:
                entity_set[entity['entity']] = text[entity['start']:entity['end']]
            elif entity == "ORG":
                entity_set["theater"] = text[entity['start']:entity['end']]
        return entity_set


    def initialize_episode(self):
        """ Refresh state for new dialog """

        nlg_model_path = self.params['nlg_model_path']
        diaact_nl_pairs = self.params['diaact_nl_pairs']
        nlg_model = nlg()
        #nlg_model.load_nlg_model(nlg_model_path)
        nlg_model.load_predefine_act_nl_pairs(diaact_nl_pairs)
        self.agent.set_nlg_model(nlg_model)
        self.cancel_check = False
        self.task_complete = False
        self.conv_over = False
        self.last_agent_action = 'greeting'
        self.last_request_slot = None

        self.agent.load_model()
        self.reward = 0
        self.episode_over = False
        self.state_tracker.initialize_episode()
        #self.state_tracker.update(user_action = self.user_action)
            
        self.agent.initialize_episode()
        self.turn += 1

        self.full_output =  open("full_output.txt", "a")
        self.full_output.write("\n\n*************************new conversation********************************")
        self.conv_output =  open("conv_output.txt", "a")
        self.conv_output.write("\n\n*************************new conversation********************************")


    def agent_next_turn(self, record_training_data=True):
        """ This function initiates each subsequent exchange between agent and user (agent first) """
        if self.fallback:
            sentence = 'I cannot understand. Can please repeat?'
        elif self.task_complete:
            self.conv_over= True
            sentence = 'Have nice day!!!!\n\n\n**************************************************************************'
        elif self.cancel_check: 
            sentence = "You are canceling the task. Thank you. Bye !!!!"
            self.conv_over = True
        elif self.state_tracker.end_conv:
            self.task_complete = True
            self.agent_action = {'act_slot_response': {'diaact': 'inform', 'inform_slots': {'taskcomplete': 'Ticket Available'}, 'request_slots': {}}, 'act_slot_value_response': None}
            sentence = self.agent.add_nl_to_taskcomplete(self.state_tracker.current_slots)


        else:
            ########################################################################
            #   CALL AGENT TO TAKE HER TURN
            ########################################################################
            self.state = self.state_tracker.get_state_for_agent()
            self.agent_action = self.agent.state_to_action(self.state)

        
            ########################################################################
            #   Register AGENT action with the state_tracker
            ########################################################################
            self.state_tracker.update(agent_action=self.agent_action)

            #self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
            sentence = self.agent.add_nl_to_action(self.agent_action)

        if not self.fallback:
            self.turn += 1
            #print("system_action : " + str(self.agent_action))

            self.last_agent_action = self.agent_action['act_slot_response']['diaact']
            self.lase_request_slot = self.agent_action['act_slot_response']['request_slots'].keys()


        if self.conv_over:
                self.full_output.close()
                self.conv_output.close()
        return (sentence, self.conv_over)


    def user_next_turn(self, sentence, record_training_data=True):
        self.fallback = False
        ########################################################################
        #   CALL USER TO TAKE HER TURN
        ########################################################################
        #self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
        #sentence = self.agent.add_nl_to_action(self.agent_action)
        self.turn += 1
        #self.user_action = ast.literal_eval(sentence)

        self.nlu_output = self.nlu.run_stack_pipelines(sentence)  
        #print("nlu_output : " + str(self.nlu_output))   
        self.user_action = self.adapter(self.nlu_output)
        if self.user_action == 'nlu_fallback':
            self.fallback = True
        if 'starttime' in self.user_action['inform_slots'] and 'critic_rating' in self.user_action['inform_slots']:
                del self.user_action['inform_slots']['critic_rating']

        #print("user_action : " + str(self.user_action))

        info_slots = ''
        for i in self.user_action['inform_slots'].keys():
            info_slots += str(i) + ' : ' + str(self.user_action['inform_slots'][i]) + ' | '

        req_slots = ''
        for i in self.user_action['request_slots'].keys():
            req_slots += str(i) + ' | '
 
        if self.print_nlu:
            print("\nIntent : " + self.user_action['diaact'] + '\nDetected informed slots : ' + info_slots + '\nDetected request slot : ' + req_slots)  
        ########################################################################
        #   Update state tracker with latest user action
        ########################################################################
        if self.episode_over != True:
            self.state_tracker.update(user_action = self.user_action)

        l= ["closing", "thanks", "thanks"]
        if self.user_action["diaact"] in l:
                self.cancel_check = True

        if self.user_action == 'deny':
                self.task_complete = False
                self.state_tracker.end_conv = False





"""
warm_start simulation episode 49: Fail
{'request_slots': {'theater': 'UNK'}, 'turn': 0, 'diaact': 'request', 'inform_slots': {'moviename': 'whiskey tango foxtrot'}}
warm_start simulation episode 50: Fail
{'request_slots': {'ticket': 'UNK'}, 'turn': 0, 'diaact': 'request', 'inform_slots': {'numberofpeople': '2', 'moviename': 'deadpool'}}
warm_start simulation episode 51: Fail
{'request_slots': {'theater': 'UNK'}, 'turn': 0, 'diaact': 'request', 'inform_slots': {'numberofpeople': '2', 'moviename': 'zootopia'}}
"""
