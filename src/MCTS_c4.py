import pickle
import os
import collections
import numpy as np
import math
import encoder_decoder_c4 as ed
from connect_board import board as c_board
import copy
import torch
import torch.multiprocessing as mp
from alpha_net_c4 import ConnectNet
import datetime
import logging
from tqdm import tqdm
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def save_as_pickle(filename, data):
    completeName = os.path.join("./data/datasets/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def load_pickle(filename):
    completeName = os.path.join("./data/datasets/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data



class UCTNode():
    def __init__(self, game, move, parent=None, c=1):
        self.game = game # state s
        self.move = move # action index
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.child_priors = np.zeros([7], dtype=np.float32)
        self.child_total_value = np.zeros([7], dtype=np.float32)
        self.child_number_visits = np.zeros([7], dtype=np.float32)
        self.child_q_mean = np.zeros([7], dtype=np.float32)
        self.child_q_m2 = np.zeros([7], dtype=np.float32)
        self.action_idxes = []
        self.c = c


    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]


    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value


    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]


    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value


    @property
    def q_mean(self):
        return self.parent.child_q_mean[self.move]


    @q_mean.setter
    def q_mean(self, value):
        self.parent.child_q_mean[self.move] = value


    @property
    def q_m2(self):
        return self.parent.child_q_m2[self.move]


    @q_m2.setter
    def q_m2(self, value):
        self.parent.child_q_m2[self.move] = value


    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)


    def child_U(self):
        return math.sqrt(self.number_visits) * (
            abs(self.child_priors) / (1 + self.child_number_visits))


    def best_child(self):
        if self.action_idxes != []:
            bestmove = self.child_Q() + self.c * self.child_U()
            bestmove = self.action_idxes[np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q() + self.c *self.child_U())
        return bestmove


    def pm_best_child(self, pm, expansions, move_history):
        action_mask = np.zeros([7], dtype=np.float32)
        action_mask[self.action_idxes] += 1
        # Generate input
        if pm.id == "SPM_base":
            inputs = np.concatenate([self.child_total_value, self.child_priors, self.child_number_visits, [self.number_visits], action_mask], dtype=np.float32)

        elif pm.id == "SPM_QVar":
            q_var = np.array([0 if self.child_number_visits[i] == 0 else self.child_q_m2[i]/self.child_number_visits[i] for i in range(7)], dtype=np.float32)
            inputs = np.concatenate([self.child_total_value, self.child_priors, self.child_number_visits, [self.number_visits], q_var, action_mask], dtype=np.float32)

        elif pm.id == "ConvPM_base":
            inputs = np.zeros([29,6,7], dtype=np.float32)
            # 22 Planes for standard PUCT inputs
            plane_idx = 0
            for d in self.child_total_value:
                inputs[plane_idx] += d / expansions   # Normalized Total Child Value
                plane_idx += 1

            for d in self.child_priors:
                inputs[plane_idx] += d                    # Child Prior
                plane_idx += 1

            for d in self.child_number_visits:
                inputs[plane_idx] += d / expansions
                plane_idx += 1

            inputs[plane_idx] += self.number_visits / expansions   # Normalized Parent Visists
            plane_idx += 1

            for d in action_mask:
                inputs[plane_idx] += d
                plane_idx += 1

        elif pm.id == "ConvPM_QVar":
            inputs = np.zeros([36,6,7], dtype=np.float32)

            # 22 Planes for standard PUCT inputs
            plane_idx = 0
            for d in self.child_total_value:
                inputs[plane_idx] += d / expansions   # Normalized Total Child Value
                plane_idx += 1

            for d in self.child_priors:
                inputs[plane_idx] += d                    # Child Prior
                plane_idx += 1

            for d in self.child_number_visits:
                inputs[plane_idx] += d / expansions
                plane_idx += 1

            inputs[plane_idx] += self.number_visits / expansions   # Normalized Parent Visists
            plane_idx += 1

            # 7 Planes for Q-Variance
            q_var = np.array([0 if self.child_number_visits[i] == 0 else self.child_q_m2[i]/self.child_number_visits[i] for i in range(7)], dtype=np.float32)
            for d in q_var:
                inputs[plane_idx] += d
                plane_idx += 1

            for d in action_mask:
                inputs[plane_idx] += d
                plane_idx += 1

        elif pm.id == "ConvPM_MH":
            inputs = np.zeros([41,6,7], dtype=np.float32)

            # 22 Planes for standard PUCT inputs
            plane_idx = 0
            for d in self.child_total_value:
                inputs[plane_idx] += d / expansions   # Normalized Total Child Value
                plane_idx += 1

            for d in self.child_priors:
                inputs[plane_idx] += d                    # Child Prior
                plane_idx += 1

            for d in self.child_number_visits:
                inputs[plane_idx] += d / expansions
                plane_idx += 1

            inputs[plane_idx] += self.number_visits / expansions   # Normalized Parent Visists
            plane_idx += 1

            # Get current histroy
            current_history = []
            current = self
            while current.move is not None:
                cboard = copy.deepcopy(current.game)
                pos = ed.encode_board(cboard).transpose(2,0,1)
                current_history.insert(0, pos)
                current = current.parent
            input_history = move_history + current_history
            if len(input_history) > 4:
                input_history = input_history[-4:]

            for pos in input_history:
                for d in pos:
                    inputs[plane_idx] += d
                    plane_idx += 1

            for i, d in enumerate(action_mask):
                inputs[34+i] += d

        elif pm.id == "ConvPM_All":
            inputs = np.zeros([48,6,7], dtype=np.float32)

            # 22 Planes for standard PUCT inputs
            plane_idx = 0
            for d in self.child_total_value:
                inputs[plane_idx] += d / expansions   # Normalized Total Child Value
                plane_idx += 1

            for d in self.child_priors:
                inputs[plane_idx] += d                    # Child Prior
                plane_idx += 1

            for d in self.child_number_visits:
                inputs[plane_idx] += d / expansions
                plane_idx += 1

            inputs[plane_idx] += self.number_visits / expansions   # Normalized Parent Visists
            plane_idx += 1

            # Get current histroy
            current_history = []
            current = self
            while current.move is not None:
                cboard = copy.deepcopy(current.game)
                pos = ed.encode_board(cboard).transpose(2,0,1)
                current_history.insert(0, pos)
                current = current.parent
            input_history = move_history + current_history
            if len(input_history) > 4:
                input_history = input_history[-4:]

            for pos in input_history:
                for d in pos:
                    inputs[plane_idx] += d
                    plane_idx += 1

            q_var = np.array([0 if self.child_number_visits[i] == 0 else self.child_q_m2[i]/self.child_number_visits[i] for i in range(7)], dtype=np.float32)
            for i, d in enumerate(q_var):
                inputs[34+i] += d

            for i, d in enumerate(action_mask):
                inputs[41+i] += d


        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        if torch.cuda.is_available():
            inputs_tensor = inputs_tensor.cuda()
        values = pm(inputs_tensor)
        values = values.detach().cpu()
        values = values.view(-1)
        if self.action_idxes != []:
            bestmove = self.action_idxes[np.argmax(values[self.action_idxes])]
        else:
            bestmove = np.argmax(values)
        return bestmove


    def select_leaf(self, generate_im_data=False, generate_ppo_data=False, pm=None, expansions=1, move_history=None):
        current = self
        data = []
        while current.is_expanded:
            best_move = current.best_child() if pm is None else current.pm_best_child(pm, expansions, move_history)
            if generate_im_data or generate_ppo_data:
                X = current.get_data(best_move)
                data.append(X)
            current = current.maybe_add_child(best_move)
        if generate_im_data or generate_ppo_data:
            return current, data
        return current


    def get_data(self, best_move):
        current = self
        move_history = []
        while current.move is not None:
            move_history.append(current.move)
            current = current.parent
        move_history.reverse()
        # Action Mask
        action_mask = np.zeros([7], dtype=np.float32)
        action_mask[self.action_idxes] += 1
        # Q value variance
        q_var = np.array([0 if self.child_number_visits[i] == 0 else self.child_q_m2[i]/self.child_number_visits[i] for i in range(7)], dtype=np.float32)
        return [copy.deepcopy(self.child_total_value), copy.deepcopy(self.child_priors), copy.deepcopy(self.child_number_visits), [copy.deepcopy(self.number_visits)], move_history, copy.deepcopy(q_var), action_mask, copy.deepcopy(self.game.player), best_move]


    def add_dirichlet_noise(self,action_idxs,child_priors):
        valid_child_priors = child_priors[action_idxs] # select only legal moves entries in child_priors array
        valid_child_priors = 0.75*valid_child_priors + 0.25*np.random.dirichlet(np.zeros([len(valid_child_priors)], \
                                                                                          dtype=np.float32)+192)
        child_priors[action_idxs] = valid_child_priors
        return child_priors


    def expand(self, child_priors):
        self.is_expanded = True
        action_idxs = self.game.actions(); c_p = child_priors
        if action_idxs == []:
            self.is_expanded = False
        self.action_idxes = action_idxs
        c_p[[i for i in range(len(child_priors)) if i not in action_idxs]] = 0.000000000 # mask all illegal actions
        if self.parent.parent == None: # add dirichlet noise to child_priors in root node
            c_p = self.add_dirichlet_noise(action_idxs,c_p)
        self.child_priors = c_p


    def decode_n_move_pieces(self,board,move):
        board.drop_piece(move)
        return board


    def maybe_add_child(self, move):
        if move not in self.children:
            copy_board = copy.deepcopy(self.game) # make copy of board
            copy_board = self.decode_n_move_pieces(copy_board,move)
            self.children[move] = UCTNode(
              copy_board, move, parent=self, c=self.c)
        return self.children[move]


    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            old_q_mean = copy.deepcopy(current.q_mean)
            q_m2 = copy.deepcopy(current.q_m2)
            n = copy.deepcopy(current.number_visits)
            if current.game.player == 1: # same as current.parent.game.player = 0
                current.total_value += (1*value_estimate) # value estimate +1 = O wins
                q_mean = old_q_mean + (value_estimate - old_q_mean) / n
                q_m2 += (value_estimate - old_q_mean) * (value_estimate - q_mean)
            elif current.game.player == 0: # same as current.parent.game.player = 1
                current.total_value += (-1*value_estimate)
                q_mean = old_q_mean + (-1*value_estimate - old_q_mean) / n
                q_m2 += (-1*value_estimate - old_q_mean) * (-1*value_estimate - q_mean)

            current.q_mean = q_mean
            current.q_m2 = q_m2
            current = current.parent



class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.child_q_mean = collections.defaultdict(float)
        self.child_q_m2 = collections.defaultdict(float)


def UCT_search(game_state, num_reads, net, temp, c=1, generate_im_data=False, planning_model=None, move_history=None, generate_ppo_data=False):
    root = UCTNode(game_state, move=None, parent=DummyNode(), c=c)
    dataset = []
    for i in range(num_reads):
        if generate_im_data:
            leaf, data = root.select_leaf(generate_im_data=generate_im_data)
            dataset.extend(data)
        elif generate_ppo_data:
            leaf, data = root.select_leaf(pm=planning_model, expansions=num_reads, move_history=move_history, generate_ppo_data=generate_ppo_data)
            dataset.extend(data)
        else:
            leaf = root.select_leaf(pm=planning_model, expansions=num_reads, move_history=move_history)
        encoded_s = ed.encode_board(leaf.game); encoded_s = encoded_s.transpose(2,0,1)
        encoded_s = torch.from_numpy(encoded_s).float().cuda()
        child_priors, value_estimate = net(encoded_s)
        child_priors = child_priors.detach().cpu().numpy().reshape(-1); value_estimate = value_estimate.item()
        if leaf.game.check_winner() == True or leaf.game.actions() == []: # if somebody won or draw
            leaf.backup(value_estimate); continue
        leaf.expand(child_priors) # need to make sure valid moves
        leaf.backup(value_estimate)
    if generate_im_data or generate_ppo_data:
        return root, dataset
    return root


def do_decode_n_move_pieces(board,move):
    board.drop_piece(move)
    return board


def get_policy(root, temp=1):
    #policy = np.zeros([7], dtype=np.float32)
    #for idx in np.where(root.child_number_visits!=0)[0]:
    #    policy[idx] = ((root.child_number_visits[idx])**(1/temp))/sum(root.child_number_visits**(1/temp))
    return ((root.child_number_visits)**(1/temp))/sum(root.child_number_visits**(1/temp))


def MCTS_self_play(connectnet, num_games, start_idx, cpu, args, iteration):
    logger.info("[CPU: %d]: Starting MCTS self-play..." % cpu)

    if not os.path.isdir("./data/datasets/iter_%d" % iteration):
        if not os.path.isdir("data/datasets"):
            os.mkdir("data/datasets")
        os.mkdir("data/datasets/iter_%d" % iteration)

    for idxx in tqdm(range(start_idx, num_games + start_idx)):
        logger.info("[CPU: %d]: Game %d" % (cpu, idxx))
        current_board = c_board()
        checkmate = False
        dataset = [] # to get state, policy, value for neural network training
        states = []
        value = 0
        move_count = 0
        while checkmate == False and current_board.actions() != []:
            if move_count < 11:
                t = args.temperature_MCTS
            else:
                t = 0.1
            states.append(copy.deepcopy(current_board.current_board))
            board_state = copy.deepcopy(ed.encode_board(current_board))
            root = UCT_search(current_board,args.expansions_per_move,connectnet,t, c=args.cpuct)
            policy = get_policy(root, t); print("[CPU: %d]: Game %d POLICY:\n " % (cpu, idxx), policy)
            current_board = do_decode_n_move_pieces(current_board,\
                                                    np.random.choice(np.array([0,1,2,3,4,5,6]), \
                                                                     p = policy)) # decode move and move piece(s)
            dataset.append([board_state,policy])
            print("[Iteration: %d CPU: %d]: Game %d CURRENT BOARD:\n" % (iteration, cpu, idxx), current_board.current_board,current_board.player); print(" ")
            if current_board.check_winner() == True: # if somebody won
                if current_board.player == 0: # black wins
                    value = -1
                elif current_board.player == 1: # white wins
                    value = 1
                checkmate = True
            move_count += 1
        dataset_p = []
        for idx,data in enumerate(dataset):
            s,p = data
            if idx == 0:
                dataset_p.append([s,p,0])
            else:
                dataset_p.append([s,p,value])
        del dataset
        save_as_pickle("iter_%d/" % iteration +\
                       "dataset_iter%d_cpu%i_%i_%s" % (iteration, cpu, idxx, datetime.datetime.today().strftime("%Y-%m-%d")), dataset_p)


def run_MCTS(args, start_idx=0, iteration=0):
    net_to_play="%s_iter%d.pth.tar" % (args.neural_net_name, iteration)
    net = ConnectNet(args.num_res_blocks)
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()

    if args.MCTS_num_processes > 1:
        logger.info("Preparing model for multi-process MCTS...")
        mp.set_start_method("spawn",force=True)
        net.share_memory()
        net.eval()

        current_net_filename = os.path.join("./data/model_data/",\
                                        net_to_play)
        if os.path.isfile(current_net_filename):
            checkpoint = torch.load(current_net_filename)
            net.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded %s model." % current_net_filename)
        else:
            if not os.path.isdir("./data/model_data/"):
                os.mkdir("data/model_data")
            torch.save({'state_dict': net.state_dict()}, os.path.join("./data/model_data/",\
                        net_to_play))
            logger.info("Initialized model.")

        processes = []
        if args.MCTS_num_processes > mp.cpu_count():
            num_processes = mp.cpu_count()
            logger.info("Required number of processes exceed number of CPUs! Setting MCTS_num_processes to %d" % num_processes)
        else:
            num_processes = args.MCTS_num_processes

        logger.info("Spawning %d processes..." % num_processes)
        with torch.no_grad():
            for i in range(num_processes):
                p = mp.Process(target=MCTS_self_play, args=(net, args.num_games_per_MCTS_process, start_idx, i, args, iteration))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        logger.info("Finished multi-process MCTS!")

    elif args.MCTS_num_processes == 1:
        logger.info("Preparing model for MCTS...")
        net.eval()

        current_net_filename = os.path.join("./data/model_data/",\
                                        net_to_play)
        if os.path.isfile(current_net_filename):
            checkpoint = torch.load(current_net_filename)
            net.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded %s model." % current_net_filename)
        else:
            if not os.path.isdir("./data/model_data/"):
                os.mkdir("data/model_data")
            torch.save({'state_dict': net.state_dict()}, os.path.join("./data/model_data/",\
                        net_to_play))
            logger.info("Initialized model.")

        with torch.no_grad():
            MCTS_self_play(net, args.num_games_per_MCTS_process, start_idx, 0, args, iteration)
        logger.info("Finished MCTS!")
