'''
Created on Oct 17, 2016

--dia_act_nl_pairs.v6.json: agt and usr have their own NL.


@author: xiul
'''

import pickle
import copy, argparse, json
import numpy as np

#import dialog_config
from deep_dialog import dialog_config
#from deep_dialog.nlg.lstm_decoder_tanh import lstm_decoder_tanh


class nlg:
    def __init__(self):
        pass
    
    def post_process(self, pred_template, slot_val_dict, slot_dict):
        """ post_process to fill the slot in the template sentence """
        
        sentence = pred_template
        suffix = "_PLACEHOLDER"
        
        for slot in slot_val_dict.keys():
            slot_vals = slot_val_dict[slot]
            slot_placeholder = slot + suffix
            if slot == 'result' or slot == 'numberofpeople': continue
            if slot_vals == dialog_config.NO_VALUE_MATCH: continue
            tmp_sentence = sentence.replace(slot_placeholder, slot_vals, 1)
            sentence = tmp_sentence
                
        if 'numberofpeople' in slot_val_dict.keys():
            slot_vals = slot_val_dict['numberofpeople']
            slot_placeholder = 'numberofpeople' + suffix
            tmp_sentence = sentence.replace(slot_placeholder, slot_vals, 1)
            sentence = tmp_sentence
    
        for slot in slot_dict.keys():
            slot_placeholder = slot + suffix
            tmp_sentence = sentence.replace(slot_placeholder, '')
            sentence = tmp_sentence
            
        return sentence


    def convert_taskcomplete_to_nl(self, dia_act):
            template_utt = "Okay, I will booked $numberofpeople$ tickets for $moviename$ starting at $starttime$ at $theater$ at $city$ on $date$."
            slots = ["numberofpeople", "moviename", "starttime", "theater", "city", "date"]
            sentence = template_utt
            for slot in slots:
                sentence = sentence.replace("$"+slot+"$",dia_act["inform_slots"][slot])
                       

            return sentence
    
    def convert_diaact_to_nl(self, dia_act, turn_msg):
        """ Convert Dia_Act into NL: Rule + Model """
        
        sentence = ""
        boolean_in = False
        
        # remove I do not care slot in task(complete)
                
        
        if dia_act['diaact'] in self.diaact_nl_pairs['dia_acts'].keys():
            for ele in self.diaact_nl_pairs['dia_acts'][dia_act['diaact']]:
                if set(ele['inform_slots']) == set(dia_act['inform_slots'].keys()) and set(ele['request_slots']) == set(dia_act['request_slots'].keys()):
                    sentence = self.diaact_to_nl_slot_filling(dia_act, ele['nl'][turn_msg])
                    boolean_in = True
                    break
        
        if dia_act['diaact'] == 'inform' and 'taskcomplete' in dia_act['inform_slots'].keys() and dia_act['inform_slots']['taskcomplete'] == dialog_config.NO_VALUE_MATCH:
            sentence = "Oh sorry, there is no ticket available."
        
        if boolean_in == False: sentence = "No sentence is available" 
        return sentence

    def diaact_to_nl_slot_filling(self, dia_act, template_sentence):
        """ Replace the slots with its values """
        
        sentence = str(template_sentence, 'utf-8')
        counter = 0
        for slot in dia_act['inform_slots'].keys():
            slot_val = dia_act['inform_slots'][slot]
            if slot_val == dialog_config.NO_VALUE_MATCH:
                sentence = "No matching " + slot + " is available!"
                break
            elif slot_val == dialog_config.I_DO_NOT_CARE:
                counter += 1
                sentence = sentence.replace('$'+slot+'$', '', 1)
                continue
            
            sentence = sentence.replace('$'+slot+'$', slot_val, 1)
        
        if counter > 0 and counter == len(dia_act['inform_slots']):
            sentence = dialog_config.I_DO_NOT_CARE
        
        return sentence
    
    
    def load_predefine_act_nl_pairs(self, path):
        """ Load some pre-defined Dia_Act&NL Pairs from file """
        
        self.diaact_nl_pairs = json.load(open(path, 'rb'))
        
        for key in self.diaact_nl_pairs['dia_acts'].keys():
            for ele in self.diaact_nl_pairs['dia_acts'][key]:
                ele['nl']['usr'] = ele['nl']['usr'].encode('utf-8') # encode issue
                ele['nl']['agt'] = ele['nl']['agt'].encode('utf-8') # encode issue


def main(params):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    params = vars(args)
    """"
    nlg = nlg()
    nlg.load_predefine_act_nl_pairs("/home/thisara/Documents/FYP/Simulator_selfplay-master/src/deep_dialog/data/dia_act_nl_pairs.v6.json")
    nlg.convert_diaact_to_nl({'diaact': 'inform', 'inform_slots': {'starttime': 'PLACEHOLDER'}, 'request_slots': {}},'agt')
    """

    main(params)
