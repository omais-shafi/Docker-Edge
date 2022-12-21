import os
import random
import traceback
import numpy as np
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.models import Model, model_from_json, load_model
from construct_sample import convert_state

import tensorflow as tf
once = True

class Agent:

    def limit_gpu(self):
        if 0:
            tf.config.gpu.set_per_process_memory_growth(True)
        elif 0:
            from keras import backend as K
            _config = tf.ConfigProto()
            _config.gpu_options.allow_growth = True
            sess = tf.Session(config=_config)
            K.setsession(sess)
        elif 1:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print('TF version', tf.__version__, 'GPUs', gpus)
            tf.config.experimental.set_memory_growth(gpus[0], True)

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id="0", isTest=False):

        global once
        if once and dic_agent_conf["GPU"]>=0:
            once = False
            self.limit_gpu()

        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.intersection_id = intersection_id
        self.dic_agent_conf = dic_agent_conf
        self.dic_path = dic_path
        self.q_teacher_network = None

        self.num_actions = len(dic_traffic_env_conf["DIC_PHASE_MAP"])-1 if not dic_traffic_env_conf["ACTION_PATTERN"] else dic_traffic_env_conf["ACTION_PATTERN"]+1
        self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))

        if self.dic_agent_conf["LOSS_FUNCTION"] != "mean_squared_error":
            raise NotImplementedError

        model_file_fmt = "round_{0}" if self.dic_traffic_env_conf['ONE_MODEL'] else "round_{0}_inter_{1}"

        if isTest:
            self.load_network(model_file_fmt.format(cnt_round, self.intersection_id))
        else:
            if self.dic_path["PATH_TO_TEACHER_MODEL"] is not None:
                self.load_teacher_network(model_file_fmt)
            if cnt_round == 0: # initialization
                if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                    self.load_network(model_file_fmt.format(0, intersection_id))
                else:
                    self.q_network = self.build_network()
                self.q_network_bar = self.build_network_from_copy(self.q_network)
            else:
                try:
                    self.load_network(model_file_fmt.format(cnt_round - 1, self.intersection_id))
                    if self.dic_path["PATH_TO_TEACHER_MODEL"] is None:
                        qbar_round = cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"]
                        self.load_network_bar(model_file_fmt.format(max(qbar_round, 0), self.intersection_id))
                except Exception as e:
                    print('traceback.format_exc():\n%s' % traceback.format_exc())

        # decay the epsilon
        if self.dic_agent_conf["EPSILON"] > 0 and self.dic_agent_conf["EPSILON_DECAY"] < 1:
            decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
            self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

    def load_teacher_network(self, model_file_fmt):
        file_path = self.dic_path["PATH_TO_TEACHER_MODEL"].split(',')[0]
        self.q_teacher_network = []
        for round in self.dic_path["PATH_TO_TEACHER_MODEL"].split(',')[1:]:
            file_name = model_file_fmt.format(round, self.intersection_id)
            self.q_teacher_network.append( load_model(os.path.join(file_path, "%s.h5" % file_name)) )
            print("succeed in loading teacher model %s"%file_name, 'at load_teacher_network')

    def load_network(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network = load_model(os.path.join(file_path, "%s.h5" % file_name))
        print("succeed in loading model %s"%file_name, 'at load_network')

    def load_network_bar(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network_bar = load_model(os.path.join(file_path, "%s.h5" % file_name))
        print("succeed in loading model %s"%file_name, 'at load_network_bar')

    def save_network(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def build_network(self):
        raise NotImplementedError

    def build_network_from_copy(self, network_copy):
        '''Initialize a Q network from a copy'''
        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure)
        network.set_weights(network_weights)
        network.compile(optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),loss=self.dic_agent_conf["LOSS_FUNCTION"])
        return network

    def prepare_Xs_Y(self, memory):

        LIST_STATE_FEATURE = "LIST_STATE_FEATURE1"
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting
        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        memory_after_forget = memory[ind_sta: ind_end]
        print("memory size after forget:", len(memory_after_forget))

        # sample the memory
        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
        sample_slice = random.sample(memory_after_forget, sample_size)
        print("memory samples number:", sample_size)

        dic_state_feature_arrays = {}
        for feature_name in self.dic_traffic_env_conf[LIST_STATE_FEATURE]:
            dic_state_feature_arrays[feature_name] = []
        Y = []

        for i in range(len(sample_slice)):
            state, action, next_state, reward, instant_reward, _, _ = sample_slice[i]

            for feature_name in self.dic_traffic_env_conf[LIST_STATE_FEATURE]:
                dic_state_feature_arrays[feature_name].append(state[feature_name])

            _state = []
            _next_state = []
            for feature_name in self.dic_traffic_env_conf[LIST_STATE_FEATURE]:
                if 1:
                    _state.append(state[feature_name])
                    _next_state.append(next_state[feature_name])
                else:
                    _state.append([state[feature_name]])
                    _next_state.append([next_state[feature_name]])
            target = self.q_network.predict(_state)

            if self.dic_path["PATH_TO_TEACHER_MODEL"] is not None:
                nw = self.q_teacher_network[ np.random.randint(len(self.q_teacher_network)) ]
                if "LIST_STATE_FEATURE2" in self.dic_traffic_env_conf.keys():
                    _state = self.convert_state_to_input(state, 'LIST_STATE_FEATURE2')
                final_target = np.copy(nw.predict(_state)[0])
            else:
                next_state_qvalues = self.q_network_bar.predict(_next_state)
                final_target = np.copy(target[0])
                final_target[action] = reward / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                       np.max(next_state_qvalues[0])
            Y.append(final_target)

        self.Xs = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in self.dic_traffic_env_conf[LIST_STATE_FEATURE]]
        self.Y = np.array(Y)

    def convert_state_to_input(self, s, feat):
        state_map = convert_state(s, self.dic_traffic_env_conf)
        inputs = []
        for feature in self.dic_traffic_env_conf[feat]:
            inputs.append(np.array([state_map[feature]]))
        return inputs

    def choose_action(self, state):

        ''' choose the best action for current state '''
        state_input = self.convert_state_to_input(state, "LIST_STATE_FEATURE1")
        q_values = self.q_network.predict(state_input)

        # Exploration exploitation
        if random.random() > self.dic_agent_conf["EPSILON"]:  # continue explore new Random Action
            action = np.argmax(q_values[0])
        elif self.q_teacher_network is None:
            action = random.randrange(len(q_values[0]))
        else:
            w = np.random.randint(len(self.q_teacher_network))
            nw = self.q_teacher_network[w]
            if "LIST_STATE_FEATURE2" in self.dic_traffic_env_conf.keys():
                state_input = self.convert_state_to_input(state, 'LIST_STATE_FEATURE2')
            q_values = nw.predict(state_input)
            action = np.argmax(q_values[0])
        return action

    def train_network(self, dic_exp_conf):

        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')
        self.q_network.fit(self.Xs, self.Y, batch_size=batch_size, epochs=self.dic_agent_conf["EPOCHS"],
                                  shuffle=False, verbose=0, validation_split=0.3, callbacks=[early_stopping])
