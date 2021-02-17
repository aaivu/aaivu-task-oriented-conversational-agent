from deep_dialog.interface import *
from deep_dialog.endtoend_system import *

def run_system(act_set, slot_set, movie_dictionary, params):
    sys_interface = interface()
    controller = ComponentController(act_set, slot_set, movie_dictionary, params)
    controller.initialize_episode()
    controller.user_next_turn("Hi")
    sys_interface.interface_setting()
    n = 0
    t = 1
    while True:
        remaining_turns = 20
        controller.initialize_episode()
        controller.user_next_turn("Hi")
        sys_interface.start_conv()
        conv_over = False
        while remaining_turns > 0 :
            user_utterance = sys_interface.user_turn()
            controller.user_next_turn(user_utterance) 
            agent_utterance, conv_over = controller.agent_next_turn()  
            sys_interface.agent_turn(agent_utterance)   
            if conv_over:
                controller.print_nlu = False
                print_nlu = input("Do you need to print nlu : ")

                if print_nlu.lower() == 'y':
                    controller.print_nlu = True
                break 
            remaining_turns -= 1
        if not conv_over:
            print("Sorry! I cannot perform the task")


