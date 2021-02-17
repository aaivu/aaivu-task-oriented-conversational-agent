import pickle
import argparse
from deep_dialog.endtoend_system import run_system

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model paths
    parser.add_argument('--trained_model_path', dest='model_path', type=str,
                        default='/home/thisara/Documents/FYP/Simulator_selfplay-master/src/new_model_1.pt', help='path to the .json dictionary file')
    parser.add_argument('--nlu_model_path', dest='nlu_model_path', type=str,
                        default='deep_dialog/data/dicts.v3.p', help='path to the .json dictionary file')
    parser.add_argument('--nlg_template_path', dest='nlg_model_path', type=str,
                        default='deep_dialog/data/dicts.v3.p', help='path to the .json dictionary file')

    # data paths
    parser.add_argument('--movie_kb_path', dest='movie_kb_path', type=str,
                        default='deep_dialog/data/movie_kb.1k.p', help='path to the movie kb .json file')

    parser.add_argument('--act_set', dest='act_set', type=str, default='deep_dialog/data/dia_acts.txt',
                        help='path to dia act set; none for loading from labeled file')
    parser.add_argument('--slot_set', dest='slot_set', type=str, default='deep_dialog/data/slot_set.txt',
                        help='path to slot set; none for loading from labeled file')
    parser.add_argument('--goal_file_path', dest='goal_file_path', type=str,
                        default='deep_dialog/data/user_goals_all_turns_template.part.movie.v1.p',
                        help='a list of user goals')
    parser.add_argument('--max_turn', dest='max_turn', default=40, type=int,
                        help='maximum length of each dialog (default=20, 0=no maximum length)')

    parser.add_argument('--diaact_nl_pairs', dest='diaact_nl_pairs', type=str,
                        default='deep_dialog/data/dia_act_nl_pairs.v6.json', help='path to the pre-defined dia_act&NL pairs')
    parser.add_argument('--nlg_model_path', dest='nlg_model_path', type=str,
                        default='deep_dialog/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p', help='path to model file')

    args = parser.parse_args()
    params = vars(args)


def text_to_dict(path):
    """ Read in a text file as a dictionary where keys are text and values are indices (line numbers) """

    slot_set = {}
    with open(path, 'r') as f:
        index = 0
        for line in f.readlines():
            slot_set[line.strip('\n').strip('\r')] = index
            index += 1
    return slot_set

movie_kb_path = params['movie_kb_path']
movie_kb = pickle.load(open(movie_kb_path, 'rb'), encoding='utf-8')
act_set = text_to_dict(params['act_set'])
slot_set = text_to_dict(params['slot_set'])

run_system(act_set, slot_set, movie_kb, params)
