import os

class interface():
    def __init__(self):
        self.agent = "DQN agent"
        self.language = 'English'
        self.agent_dic = {1: "DQN agent", 2: "Double agent", 3: "Duel agent",
                          4: "NoisyNet DQN agent", 5: "Prioritized DQN agent"}
        self.lan_dic = {'e': 'English', 's': 'Sinhala'}

    def interface_setting(self):
        print("****************************************************************\n"
              "Hi !!! Welcome to the movie booking chatbot ....\n"
              "****************************************************************\n")
        print('Language : ' + self.language + "\nRL agent : " + self.agent)
        change = input("\nDo you want to change the settings?(y)")
        change = change.lower()
        if change == 'y' or change == 'yes':
            print("\nLanguage setting : L\nRL agent setting : A")
            self.change_setting()

    def change_setting(self):
        repeat = True
        while repeat:
            setting = input("\nPlease enter the code for required setting >> ")
            setting = setting.lower()
            if setting == 'l':
                self.select_language()
                out = input("Do you want to do more setting changes?(y)")
                out = out.lower()
                if not (out == 'y' or out == 'yes'):
                    repeat = False
            elif setting == 'a':
                self.select_agent()
                out = input("Do you want to do more setting changes?(y)")
                out = out.lower()
                if not (out == 'y' or out == 'yes'):
                    repeat = False
            else:
                print('\nNot matching code for any valid setting')
                out = input("Do you want change the settings? (y)")
                out = out.lower()
                if not (out == 'y' or out == 'yes'):
                    repeat = False

    def select_language(self):
        print("----------------------------------------------------------------\n"
              "Language selection\n")
        for i in list(self.lan_dic.keys()):
            print ("\t" + self.lan_dic[i] + " : " + str(i))
        repeat = True
        while repeat:
            language = input("\nPlease enter the code for preferred language >> ")
            language = language.lower()
            if language == 'e' or language == 'english':
                self.language = 'english'
                print("\nSuccessfully change the language to the English\n")
                repeat = False
            elif language == 's' or language == 'sinhala':
                self.language = 'sinhala'
                print("\nSuccessfully change the language to the Sinhala\n")
                repeat = False
            else:
                print('\nNot matching code for any valid language')
                out = input("Do you want change the language? (y)")
                out = out.lower()
                if not (out == 'y' or out == 'yes'):
                    repeat = False
        print("----------------------------------------------------------------\n")

    def select_agent(self):
        print("----------------------------------------------------------------\n"
              "RL agent selection\n")
        for i in list(self.agent_dic.keys()):
            print ("\t" + self.agent_dic[i] + " : " + str(i))
        repeat = True
        while repeat:
            agent = input("\nPlease enter the code for preferred RL agent >> ")
            try:
                agt = int(agent)
                if agt in self.agent_dic:
                    self.agent = self.agent_dic[agt]
                    repeat = False
                else:
                    print('\nNot matching code for any valid RL agent')
                    out = input("Do you want change the RL agent? (y)")
                    out = out.lower()
                    if not (out == 'y' or out == 'yes'):
                        repeat = False
            except ValueError:
                print("\ninvalid input. Must be a number")
                out = input("Do you want change the RL agent? (y)")
                out = out.lower()
                if not (out == 'y' or out == 'yes'):
                    repeat = False
        print("----------------------------------------------------------------\n")

    def start_conv(self):
        os.system("clear")
        print("\n\n\n****************************************************************\n"
              "Chatbot >>> Hi !! How Can I help you?\n")

    def user_turn(self):
        user_utterance = ''
        while user_utterance.strip() == '':
            user_utterance = input(">>> ")
        return user_utterance

    def agent_turn(self, agent_utternace):
        print("\nChatbot >>> " + agent_utternace + '\n')



