import os
import copy
import time
from config import DIC_AGENTS, DIC_ENVS

class Generator:
    def __init__(self, cnt_round, cnt_gen, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, isTest=False):

        self.isTest = isTest
        self.dic_exp_conf = dic_exp_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.num_agents = 1 if dic_traffic_env_conf['SINGLE_AGENT'] else dic_traffic_env_conf["NUM_AGENTS"]

        self.path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round" if isTest else "train_round", "round_"+str(cnt_round), "generator_"+str(cnt_gen))
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)

        start_time = time.time()
        agent_name = self.dic_exp_conf["MODEL_NAME"]
        self.agents = []
        for i in range(self.num_agents):
            self.agents.append(DIC_AGENTS[agent_name](
                                    dic_agent_conf=dic_agent_conf,
                                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                                    dic_path=dic_path,
                                    cnt_round=cnt_round,
                                    intersection_id=str(i),
                                    isTest=isTest
                                ))
        print("Create intersection agent time: ", time.time()-start_time)

        self.env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
                                path_to_log = self.path_to_log,
                                path_to_work_directory = dic_path["PATH_TO_WORK_DIRECTORY"],
                                dic_traffic_env_conf = self.dic_traffic_env_conf)

    def merge_states(self, state):
        st = copy.deepcopy(state[0])
        for i in range(1, self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            for key, value in st.items():
                st[key].extend(state[i][key])
        return st

    def generate(self):

        reset_env_start_time = time.time()
        state = self.env.reset()
        reset_env_time = time.time() - reset_env_start_time
        running_start_time = time.time()

        done = False
        step_num = 0
        while not done and step_num < int(self.dic_exp_conf["RUN_COUNTS"]/self.dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []
            for i in range(self.num_agents):
                one_state = state[i] if not self.dic_traffic_env_conf['SINGLE_AGENT'] else self.merge_states(state)
                action_list.append([self.agents[i].choose_action(one_state)])

            next_state, reward, done = self.env.step(action_list,False)
            state = next_state
            step_num += 1

        running_time = time.time() - running_start_time
        log_start_time = time.time()
        print("start logging")
        self.env.bulk_log_multi_process()
        log_time = time.time() - log_start_time

        if self.isTest:
            self.env.log_metrics()

        self.env.end_anon()
        print("reset_env_time: ", reset_env_time)
        print("running_time: ", running_time)
        print("log_time: ", log_time)
