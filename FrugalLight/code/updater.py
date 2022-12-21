import os
import time
import pickle
import traceback
from config import DIC_AGENTS

class Updater:

    def __init__(self, cnt_round, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path):

        self.cnt_round = cnt_round
        self.dic_path = dic_path
        self.dic_exp_conf = dic_exp_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        self.agents = []
        self.sample_set_list = []
        self.sample_indexes = None
        self.num_agents = 1 if dic_traffic_env_conf['SINGLE_AGENT'] else dic_traffic_env_conf["NUM_AGENTS"]

        agent_name = self.dic_exp_conf["MODEL_NAME"]
        for i in range(self.num_agents):
            agent = DIC_AGENTS[agent_name](
                    self.dic_agent_conf, self.dic_traffic_env_conf, self.dic_path,
                    self.cnt_round, intersection_id=str(i))
            self.agents.append(agent)

    def load_sample_with_forget(self, i):

        sample_set = []
        try:
            sample_file = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                            "total_samples_inter_{0}.pkl".format(i)), "rb")
            try:
                while True:
                    sample_set += pickle.load(sample_file)
                    ind_end = len(sample_set)
                    ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
                    memory_after_forget = sample_set[ind_sta: ind_end]
                    sample_set = memory_after_forget

            except EOFError:
                pass
        except Exception as e:
            error_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"]).replace("records", "errors")
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            f = open(os.path.join(error_dir, "error_info_inter_{0}.txt".format(i)), "a")
            f.write("Fail to load samples for inter {0}\n".format(i))
            f.write('traceback.format_exc():\n%s\n' % traceback.format_exc())
            f.close()
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            pass
        if i % 100 == 0:
            print("load_sample for inter {0}".format(i))
        return sample_set

    def load_sample_for_agents(self):
        start_time = time.time()
        print("Start load samples at", start_time)
        for i in range(self.num_agents):
            sample_set = self.load_sample_with_forget(i)
            self.agents[i].prepare_Xs_Y(sample_set)
        print("------------------Load samples time: ", time.time()-start_time)

    def update_network(self,i):
        print('update agent %d'%(i))
        self.agents[i].train_network(self.dic_exp_conf)
        if self.dic_traffic_env_conf["ONE_MODEL"]:
            self.agents[i].save_network("round_{0}".format(self.cnt_round))
        else:
            self.agents[i].save_network("round_{0}_inter_{1}".format(self.cnt_round,self.agents[i].intersection_id))

    def update_network_for_agents(self):
        if self.dic_traffic_env_conf["ONE_MODEL"]:
            self.update_network(0)
        else:
            print("update_network_for_agents", self.num_agents)
            for i in range(self.num_agents):
                self.update_network(i)
