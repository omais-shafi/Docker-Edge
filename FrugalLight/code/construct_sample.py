import os
import copy
import pickle
import traceback
import numpy as np

def convert_state(state, dic_traffic_env_conf):
    return {key: value for key, value in state.items() if key in dic_traffic_env_conf["LIST_STATE_FEATURE"]}

class ConstructSample:

    def __init__(self, path_to_samples, cnt_round, dic_traffic_env_conf):
        self.parent_dir = path_to_samples
        self.path_to_samples = path_to_samples + "/round_" + str(cnt_round)
        self.cnt_round = cnt_round
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.logging_data_list_per_gen = None
        self.hidden_states_list = None
        self.samples_all_intersection = [None]*self.dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_agents = 1 if dic_traffic_env_conf['SINGLE_AGENT'] else dic_traffic_env_conf["NUM_INTERSECTIONS"]

    def load_data(self, folder, i):
        try:
            f_logging_data = open(os.path.join(self.path_to_samples, folder, "inter_{0}.pkl".format(i)), "rb")
            logging_data = pickle.load(f_logging_data)
            f_logging_data.close()
            return 1, logging_data
        except Exception as e:
            print("Error occurs when making samples for inter {0}".format(i))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0, None

    def load_data_for_system(self, folder):
        '''
        Load data for all intersections in one folder
        :param folder:
        :return: a list of logging data of one intersection for one folder
        '''
        print("Load data for system in ", folder)
        self.logging_data_list_per_gen = []
        self.measure_time = self.dic_traffic_env_conf["MEASURE_TIME"]
        self.interval = self.dic_traffic_env_conf["MIN_ACTION_TIME"]

        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            pass_code, logging_data = self.load_data(folder, i)
            if pass_code == 0:
                return 0
            self.logging_data_list_per_gen.append(logging_data)
        return 1

    def load_hidden_state_for_system(self, folder):
        print("loading hidden states: {0}".format(os.path.join(self.path_to_samples, folder, "hidden_states.pkl")))
        if self.hidden_states_list is None:
            self.hidden_states_list = []

        try:
            f_hidden_state_data = open(os.path.join(self.path_to_samples, folder, "hidden_states.pkl"), "rb")
            hidden_state_data = pickle.load(f_hidden_state_data) # hidden state_data is a list of numpy array
            self.hidden_states_list.append(np.stack(hidden_state_data, axis=2))
            return 1
        except Exception as e:
            print("Error occurs when loading hidden states in ", folder)
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0

    def construct_state(self,time,i):
        state = self.logging_data_list_per_gen[i][time]
        assert time == state["time"]
        return convert_state(state["state"], self.dic_traffic_env_conf)

    def get_reward_from_features(self, rs):
        reward = {}
        reward["xcnt"] = np.sum(rs["num_vehicle_left"])
        reward['mp']   = np.absolute(np.sum(rs["pressure"]))
        reward["stop"] = np.sum(rs["vehicles_been_stopped_thres1"])
        reward["qden"] = np.sum(rs["transform_approach"])
        return reward

    def cal_reward(self, rs, rewards_components):
        r = 0
        for component, weight in rewards_components.items():
            if weight == 0:
                continue
            if component not in rs.keys():
                continue
            if rs[component] is None:
                continue
            r += rs[component] * weight
        return r

    def construct_reward(self,rewards_components,time, i):
        rs = self.logging_data_list_per_gen[i][time + self.measure_time - 1]
        assert time + self.measure_time - 1 == rs["time"]
        rs = self.get_reward_from_features(rs['state'])
        r_instant = self.cal_reward(rs, rewards_components)

        # average
        list_r = []
        for t in range(time, time + self.measure_time):
            rs = self.logging_data_list_per_gen[i][t]
            assert t == rs["time"]
            rs = self.get_reward_from_features(rs['state'])
            r = self.cal_reward(rs, rewards_components)
            list_r.append(r)
        r_average = np.average(list_r)

        return r_instant, r_average

    def judge_action(self,time,i):
        if self.logging_data_list_per_gen[i][time]['action'] == -1:
            raise ValueError
        return self.logging_data_list_per_gen[i][time]['action']

    def make_reward(self, folder, i):
        '''
        make reward for one folder and one intersection,
        add the samples of one intersection into the list.samples_all_intersection[i]
        :param i: intersection id
        '''
        if self.samples_all_intersection[i] is None:
            self.samples_all_intersection[i] = [[]]

        if i % 100 == 0:
            print("make reward for inter {0} in folder {1}".format(i, folder))

        try:
            total_time = int(self.logging_data_list_per_gen[i][-1]['time'] + 1)
            # construct samples
            list_samples = []
            for time in range(0, total_time - self.measure_time + 1, self.interval):
                state = self.construct_state(time, i)
                action = self.judge_action(time, i)
                next_state = self.construct_state(time + self.interval - (1 if time + self.interval == total_time else 0), i)

                rew_instant, rew_avg = self.construct_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"], time, i)
                sample = [state, action, next_state, rew_avg, rew_instant, time, folder+"-"+"round_{0}".format(self.cnt_round)]
                list_samples.append(sample)

            self.samples_all_intersection[i][0].extend(list_samples)
            return 1
        except Exception as e:
            print("Error occurs when making rewards in generator {0} for intersection {1}".format(folder, i))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0

    def gen_single_states(self):
        merged_samples = []
        for k in range(len(self.samples_all_intersection[0][0])):
            state, action, next_state, reward_average, reward_instant, time, folder = self.samples_all_intersection[0][0][k]
            state = copy.deepcopy(state)
            next_state = copy.deepcopy(next_state)
            action = [action]
            reward_average = [reward_average]
            for i in range(1,self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
                state_, action_, next_state_, reward_average_,_,_,_ = self.samples_all_intersection[i][0][k]
                reward_average.append(reward_average_)
                action.append(action_)
                for key, value in state.items():
                    state[key].extend(state_[key])
                    next_state[key].extend(next_state_[key])
            sample = [state, action, next_state, reward_average, reward_instant, time, folder]
            merged_samples.append(sample)

        return merged_samples

    # Entry func
    def make_reward_for_system(self):
        '''
        Iterate all the generator folders, and load all the logging data for all intersections for that folder
        At last, save all the logging data for that intersection [all the generators]
        '''
        for folder in os.listdir(self.path_to_samples):
            if "generator" in folder and self.load_data_for_system(folder):
                for i in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
                    self.make_reward(folder, i)

        if self.dic_traffic_env_conf['SINGLE_AGENT']:
            new_samples = self.gen_single_states()
            self.dump_sample(new_samples,"inter_0")
        else:
            for i in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
                self.dump_sample(self.samples_all_intersection[i][0],"inter_{0}".format(i))

    def dump_sample(self, samples, folder):
        if folder == "":
            f = open(os.path.join(self.parent_dir, "total_samples.pkl"),"ab+")
        elif "inter" in folder:
            f = open(os.path.join(self.parent_dir, "total_samples_{0}.pkl".format(folder)),"ab+")
        else:
            f = open(os.path.join(self.path_to_samples, folder, "samples_{0}.pkl".format(folder)),'wb')

        pickle.dump(samples, f, -1)
        f.close()
