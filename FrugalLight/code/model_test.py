import json
import os
from config import DIC_AGENTS, DIC_ENVS
from copy import deepcopy
import traceback
import argparse

def load_conf(records_dir):
    dic_path = {}
    dic_path["PATH_TO_MODEL"] = os.path.join(records_dir, 'model')
    dic_path["PATH_TO_WORK_DIRECTORY"] = records_dir

    with open(os.path.join(records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)
    with open(os.path.join(records_dir, "exp.conf"), "r") as f:
        dic_exp_conf = json.load(f)
    with open(os.path.join(records_dir, "traffic_env.conf"), "r") as f:
        dic_traffic_env_conf = json.load(f)

    return dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path


def load_agents(dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path, cnt_round=1):
    agents = []
    agent_name = dic_exp_conf["MODEL_NAME"]
    for i in range(dic_traffic_env_conf['NUM_AGENTS']):
        agents.append(DIC_AGENTS[agent_name](
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            cnt_round=cnt_round,  # useless
            intersection_id=str(i),
            isTest=True
        ))
    return agents


def update_conf(dic_path, dic_traffic_env_conf, dic_agent_conf):
    import config
    # Convert MAPS
    if 'DIC_PHASE_GREEN_APPROACH_MAP' in dic_traffic_env_conf.keys():
        map = dic_traffic_env_conf["DIC_PHASE_GREEN_APPROACH_MAP"]
        map2 = {}
        for key,val in map.items():
            map2[int(key)] = val
        dic_traffic_env_conf["DIC_PHASE_GREEN_APPROACH_MAP"] = map2
    else:
        dic_traffic_env_conf["DIC_PHASE_GREEN_APPROACH_MAP"] = config.DIC_PHASE_GREEN_APPROACH_MAP

    if 'DIC_PHASE_MAP' in dic_traffic_env_conf.keys():
        map = dic_traffic_env_conf["DIC_PHASE_MAP"]
        map2 = {}
        for key,val in map.items():
            map2[int(key)] = val
        dic_traffic_env_conf["DIC_PHASE_MAP"] = map2
    else:
        dic_traffic_env_conf["DIC_PHASE_MAP"] = config.DIC_PHASE_MAP_CYCLIC

def test(records_dir, model_dir, suffix, cnt_round_start, cnt_round_end, run_cnt=0, _dic_traffic_env_conf=None, test_only=False):

    dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path = load_conf(records_dir)
    if _dic_traffic_env_conf is not None:
        dic_traffic_env_conf = deepcopy(_dic_traffic_env_conf)
    if test_only:
        update_conf(dic_path, dic_traffic_env_conf, dic_agent_conf)

    if run_cnt > 0:
        dic_exp_conf["RUN_COUNTS"] = run_cnt

    # dump dic_exp_conf
    with open(os.path.join(records_dir, "test_exp.conf"), "w") as f:
        json.dump(dic_exp_conf, f)

    if dic_exp_conf["MODEL_NAME"] in dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"] or dic_traffic_env_conf["TRANSFER"]:
        dic_agent_conf["EPSILON"] = 0
        dic_agent_conf["MIN_EPSILON"] = 0

    total_time = dic_exp_conf["RUN_COUNTS"]
    dic_traffic_env_conf["RUN_COUNTS"] = total_time

    try:
        for cnt_round in range(cnt_round_start, cnt_round_end+1):

            agents = load_agents(dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path, cnt_round)
            model_round = "round_%d" % cnt_round
            path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round"+suffix, model_round)
            if not os.path.exists(path_to_log):
                os.makedirs(path_to_log)
            env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](path_to_log=path_to_log,
                                                                   path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
                                                                   dic_traffic_env_conf=dic_traffic_env_conf)
            done = False
            step_num = 0
            state = env.reset()

            while not done and step_num < int(total_time / dic_traffic_env_conf["MIN_ACTION_TIME"]):
                action_list = []
                for i in range(dic_traffic_env_conf["NUM_AGENTS"]):
                    one_state = state[i]
                    act = agents[i].choose_action(one_state)
                    action_list.append([act])

                next_state, reward, done = env.step(action_list, False)
                state = next_state
                step_num += 1

            env.log_metrics()

    except:
        print('traceback.format_exc():\n%s' % traceback.format_exc())
        error_dir = model_dir.replace("model", "errors")
        if not os.path.exists(error_dir):
            os.makedirs(error_dir)
        with open(os.path.join(error_dir, "error_info.txt"), "a") as f:
            f.write("round_%d traceback.format_exc():\n%s" % (cnt_round,traceback.format_exc()))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if args.start > args.end:
        print('Provide valid start end')
    else:
        records_dir = os.path.join('records/test', args.dir) if '/' not in args.dir else args.dir
        test(records_dir, os.path.join(records_dir, 'model'), '_', args.start, args.end, test_only=True)
