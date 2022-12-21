import os
import time
import json
import shutil
import random
import pickle
import traceback
from copy import deepcopy
from multiprocessing import Process

import model_test
from updater import Updater
from generator import Generator
from construct_sample import ConstructSample

class Pipeline:

    def _copy_conf_file(self, path=None):
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        json.dump(self.dic_exp_conf, open(os.path.join(path, "exp.conf"), "w"), indent=4)
        json.dump(self.dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"), indent=4)
        json.dump(self.dic_traffic_env_conf, open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)

    def _copy_anon_file(self, path=None):
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_exp_conf["TRAFFIC_FILE"][0]),
                    os.path.join(path, self.dic_exp_conf["TRAFFIC_FILE"][0]))
        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_traffic_env_conf["TRAFFIC_FILE"]),
                    os.path.join(path, self.dic_traffic_env_conf["TRAFFIC_FILE"]))
        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_exp_conf["ROADNET_FILE"]),
                    os.path.join(path, self.dic_exp_conf["ROADNET_FILE"]))

    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):

        # load configurations
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        # do file operations
        self._copy_conf_file()
        self._copy_anon_file()
        self.test_duration = []

        sample_num = 10 if self.dic_traffic_env_conf["NUM_INTERSECTIONS"]>=10 else min(self.dic_traffic_env_conf["NUM_INTERSECTIONS"], 9)
        print("sample_num for early stopping:", sample_num)
        self.sample_inter_id = random.sample(range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]), sample_num)
        self.no_test = dic_traffic_env_conf['NO_TEST'] if 'NO_TEST' in dic_traffic_env_conf.keys() else 0

        self.dic_agent_conf_test = deepcopy(self.dic_agent_conf)
        self.dic_agent_conf_test["EPSILON"] = 0
        self.dic_agent_conf_test["MIN_EPSILON"] = 0

    def generator_wrapper(self, cnt_round, cnt_gen, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, isTest=False):
        generator = Generator(cnt_round=cnt_round,
                              cnt_gen=cnt_gen,
                              dic_path=dic_path,
                              dic_exp_conf=dic_exp_conf,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=dic_traffic_env_conf,
                              isTest=isTest
                              )
        print("make generator")
        generator.generate()
        print("generator_wrapper end")

    def updater_wrapper(self, cnt_round, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path):
        updater = Updater(cnt_round=cnt_round,
                          dic_agent_conf=dic_agent_conf,
                          dic_exp_conf=dic_exp_conf,
                          dic_traffic_env_conf=dic_traffic_env_conf,
                          dic_path=dic_path
                          )
        updater.load_sample_for_agents()
        updater.update_network_for_agents()
        print("updater_wrapper end")

    def downsample(self, path_to_log, i):
        path_to_pkl = os.path.join(path_to_log, "inter_{0}.pkl".format(i))
        with open(path_to_pkl, "rb") as f_logging_data:
            try:
                logging_data = pickle.load(f_logging_data)
                subset_data = logging_data[::10]
                os.remove(path_to_pkl)
                with open(path_to_pkl, "wb") as f_subset:
                    try:
                        pickle.dump(subset_data, f_subset)
                    except Exception as e:
                        print("Error occurs when WRITING pickles when down sampling for inter {0}".format(i))
                        print('traceback.format_exc():\n%s' % traceback.format_exc())
            except Exception as e:
                print("Error occurs when READING pickles when down sampling for inter {0}".format(i))
                print('traceback.format_exc():\n%s' % traceback.format_exc())

    def downsample_for_system(self, path_to_log, dic_traffic_env_conf):
        for i in range(dic_traffic_env_conf['NUM_INTERSECTIONS']):
            self.downsample(path_to_log, i)

    def construct_sample_multi_process(self, train_round, cnt_round, batch_size=200):
        cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                             dic_traffic_env_conf=self.dic_traffic_env_conf)
        if batch_size > self.dic_traffic_env_conf['NUM_INTERSECTIONS']:
            batch_size_run = self.dic_traffic_env_conf['NUM_INTERSECTIONS']
        else:
            batch_size_run = batch_size
        process_list = []
        for batch in range(0, self.dic_traffic_env_conf['NUM_INTERSECTIONS'], batch_size_run):
            start = batch
            stop = min(batch + batch_size, self.dic_traffic_env_conf['NUM_INTERSECTIONS'])
            process_list.append(Process(target=self.construct_sample_batch, args=(cs, start, stop)))

        for t in process_list:
            t.start()
        for t in process_list:
            t.join()

    def construct_sample_batch(self, cs, start,stop):
        for inter_id in range(start, stop):
            print("make construct_sample_wrapper for ", inter_id)
            cs.make_reward(inter_id)

    def generate_samples(self, cnt_round, multi_process, isTest=True, num_gen=1):
        process_list = []
        generator_start_time = time.time()
        dic_agent_conf = self.dic_agent_conf_test if isTest else self.dic_agent_conf

        if multi_process:
            for cnt_gen in range(num_gen):
                p = Process(target=self.generator_wrapper,
                            args=(cnt_round, cnt_gen, self.dic_path, self.dic_exp_conf,
                                  dic_agent_conf, self.dic_traffic_env_conf, isTest)
                            )
                print("before")
                p.start()
                print("end")
                process_list.append(p)
            print("before join")
            for i in range(len(process_list)):
                p = process_list[i]
                print("generator %d to join" % i)
                p.join()
                print("generator %d finish join" % i)
            print("end join")
        else:
            for cnt_gen in range(num_gen):
                self.generator_wrapper(cnt_round=cnt_round,
                                       cnt_gen=cnt_gen,
                                       dic_path=self.dic_path,
                                       dic_exp_conf=self.dic_exp_conf,
                                       dic_agent_conf=dic_agent_conf,
                                       dic_traffic_env_conf=self.dic_traffic_env_conf,
                                       isTest=isTest)
        generator_total_time = time.time() - generator_start_time
        print("==============  make samples =============")
        # make samples and determine which samples are good
        making_samples_start_time = time.time()
        round_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "test_round" if isTest else "train_round")
        if not os.path.exists(round_dir):
            os.makedirs(round_dir)
        cs = ConstructSample(path_to_samples=round_dir, cnt_round=cnt_round,
                             dic_traffic_env_conf=self.dic_traffic_env_conf)
        cs.make_reward_for_system()

        making_samples_total_time = time.time() - making_samples_start_time
        return generator_total_time, making_samples_total_time

    def run(self, multi_process=False):
        f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"],"running_time.csv"),"w")
        f_time.write("generator_time\tmaking_samples_time\tupdate_network_time\ttest_evaluation_times\tall_times\n")
        f_time.close()

        cnt_round = self.dic_exp_conf["START_ROUNDS"]-1
        while cnt_round < self.dic_exp_conf["START_ROUNDS"]+self.dic_exp_conf["NUM_ROUNDS"]-1:
            cnt_round += 1
            print("round %d starts" % cnt_round)
            round_start_time = time.time()

            if self.dic_exp_conf["MODEL_NAME"] in self.dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
                print("==============  generator =============")
                generator_total_time, making_samples_total_time = self.generate_samples(cnt_round, multi_process, False, self.dic_exp_conf["NUM_GENERATORS"])

                print("==============  update network =============")
                update_network_start_time = time.time()
                if multi_process:
                    p = Process(target=self.updater_wrapper,
                                args=(cnt_round,
                                      self.dic_agent_conf,
                                      self.dic_exp_conf,
                                      self.dic_traffic_env_conf,
                                      self.dic_path
                                      ))
                    p.start()
                    print("update to join")
                    p.join()
                    print("update finish join")
                else:
                    self.updater_wrapper(cnt_round=cnt_round,
                                         dic_agent_conf=self.dic_agent_conf,
                                         dic_exp_conf=self.dic_exp_conf,
                                         dic_traffic_env_conf=self.dic_traffic_env_conf,
                                         dic_path=self.dic_path)

                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                               "round_" + str(cnt_round), "generator_" + str(cnt_gen))
                    self.downsample_for_system(path_to_log,self.dic_traffic_env_conf)
                update_network_end_time = time.time()
                update_network_total_time = update_network_end_time - update_network_start_time

            test_evaluation_total_time = 0
            if self.no_test < 0:
                print("==============  test evaluation =============")
                test_evaluation_start_time = time.time()
                if multi_process:
                    p = Process(target=model_test.test,
                                args=(self.dic_path["PATH_TO_WORK_DIRECTORY"], self.dic_path["PATH_TO_MODEL"], '', cnt_round, cnt_round, self.dic_exp_conf["RUN_COUNTS"], self.dic_traffic_env_conf, False))
                    p.start()
                else:
                    model_test.test(self.dic_path["PATH_TO_WORK_DIRECTORY"], self.dic_path["PATH_TO_MODEL"], '', cnt_round, cnt_round, self.dic_exp_conf["RUN_COUNTS"], self.dic_traffic_env_conf)
                    test_evaluation_total_time = time.time() - test_evaluation_start_time
            elif cnt_round >= self.no_test:
                generator_total_time2, making_samples_total_time2 = self.generate_samples(cnt_round, multi_process)
                test_evaluation_total_time = generator_total_time2 + making_samples_total_time2

            if "generator_total_time" in locals():
                print("Generator time:", generator_total_time)
                print("Making samples time:", making_samples_total_time)
                print("update_network time:", update_network_total_time)
            print("test_evaluation time:", test_evaluation_total_time)

            print("round {0} ends, total_time: {1}".format(cnt_round, time.time()-round_start_time))

            if "generator_total_time" in locals():
                f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"],"running_time.csv"),"a")
                f_time.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(generator_total_time,making_samples_total_time,
                                                              update_network_total_time,test_evaluation_total_time,
                                                              time.time()-round_start_time))
                f_time.close()
