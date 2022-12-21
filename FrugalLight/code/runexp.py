import os
import sys
import copy
import time
import random
import argparse
from pipeline import Pipeline
import config
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transform", type=str, default='Lane')
    parser.add_argument("--road_net", type=str, default='1_1_Sym300')
    parser.add_argument("--density", type=int, default=39)
    parser.add_argument("--rewards", type=str, default='stop')
    parser.add_argument("--action_pattern", type=int, default=1)    # switch, just 1 next
    parser.add_argument("--phasetype", type=str, default='cyclic')

    parser.add_argument("--num_phase", type=int, default=4)
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--memo", type=str, default='test')
    parser.add_argument("--mod", type=str, default="DQN")
    parser.add_argument("--lane", type=int, default=3)

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dense", type=str, default=None)
    parser.add_argument("--xfer", type=str, default=None)
    parser.add_argument("--traffic", type=int, default=1)
    parser.add_argument("--dir_suffix", type=str, default=None)
    parser.add_argument("-onemodel", action="store_true", default=False)
    parser.add_argument("-oneprocess", action="store_true", default=False)
    parser.add_argument("-singleagent", action="store_true", default=False)
    parser.add_argument("-dryrun", action="store_true", default=False)
    parser.add_argument("--msg", type=str, default=None)
    parser.add_argument("--notest", type=int, default=0)

    parser.add_argument("--teach", type=str, default=None)
    parser.add_argument("-tblind", action="store_true",default=False) # or explored by default
    parser.add_argument("--twho", type=str, default=None) # If teacher a different model

    parser.add_argument("--cnt",type=int, default=3600)
    parser.add_argument("--gen",type=int, default=1)
    parser.add_argument("--workers",type=int, default=8)
    parser.add_argument("--gpu", type=int, default=-1)
    return parser.parse_args()

def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result

def pipeline_wrapper(multi_process, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
    ppl = Pipeline(dic_exp_conf=dic_exp_conf,
                   dic_agent_conf=dic_agent_conf,
                   dic_traffic_env_conf=dic_traffic_env_conf,
                   dic_path=dic_path
                   )
    ppl.run(multi_process=multi_process)
    print("pipeline_wrapper end")

def main(args = None):

    dense_map = {'lane': '20', 'approach': '15', 'group': '10', 'relative': '5'}
    for k in dense_map.keys():
        if k.startswith(args.transform.lower()):
            args.transform = k
            break
    if args.transform not in dense_map.keys():
        print('Valid transform parameter (Lane,Approach,Group,Relative) is must!!!')
        exit(0)
    if args.dense is None:
        args.dense = dense_map[args.transform]

    ENVIRONMENT = "anon"
    if args.road_net.startswith("16_1"):
        traffic_file_list = ["anon_16_1_newyork_real_{0}.json".format(args.traffic)]
    elif args.road_net.startswith("16_3"):
        traffic_file_list = ["anon_16_3_newyork_real_{0}.json".format(args.traffic)]
    elif args.road_net.startswith("1_1"):
        traffic_file_list = ["anon_1_1_delhi_real_{0}.json".format(args.traffic)]
        args.num_phase = 3
    else:
        traffic_file_list = ["{0}_{1}.json".format(ENVIRONMENT, args.road_net)]

    list_roadnet = args.road_net.split('_')
    NUM_COL = int(list_roadnet[0])
    NUM_ROW = int(list_roadnet[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)
    print('traffic_file_list', traffic_file_list)

    for traffic_file in traffic_file_list:

        dic_exp_conf_extra = {
            "RUN_COUNTS": args.cnt,
            "MODEL_NAME": args.mod,
            "TRAFFIC_FILE": [traffic_file], # here: change to multi_traffic
            "ROADNET_FILE": "roadnet_{0}.json".format(args.road_net),
            "NUM_ROUNDS": args.num_rounds,
            "NUM_GENERATORS": args.gen,
        }

        dic_agent_conf_extra = {
            "N_LAYER": args.num_layers,
            "TRAFFIC_FILE": traffic_file,
        }

        dic_traffic_env_conf_extra = {
            "ONE_MODEL": args.onemodel,
            "NUM_AGENTS": num_intersections,
            "NUM_INTERSECTIONS": num_intersections,
            "ACTION_PATTERN": args.action_pattern,
            "RUN_COUNTS": args.cnt,
            "SIMULATOR_TYPE": ENVIRONMENT,
            "MODEL_NAME": args.mod,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,
            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": "roadnet_{0}.json".format(args.road_net),
            "TRAFFIC_SEPARATE": traffic_file,
            "DIC_PHASE_GREEN_APPROACH_MAP": config.DIC_PHASE_GREEN_APPROACH_MAP,
            "DIC_PHASE_MAP": config.DIC_PHASE_MAP_CYCLIC,
        }

        if args.resume is None:
            suffix = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))+'_'+(args.dir_suffix if args.dir_suffix is not None else str(random.randint(1,100)))
        else:
            resume_list = args.resume.split(',')
            suffix = resume_list[0]
            dic_exp_conf_extra["START_ROUNDS"] = int(resume_list[1])
            dic_exp_conf_extra["NUM_ROUNDS"] -= dic_exp_conf_extra["START_ROUNDS"]

        dic_path_extra = {
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", args.memo, traffic_file + "_" + suffix),
            "PATH_TO_MODEL": os.path.join("records", args.memo, traffic_file + "_" + suffix, 'model'),
            "PATH_TO_TRANSFER_MODEL":os.path.join("data",args.road_net),
            "PATH_TO_DATA": os.path.join("data", list_roadnet[0]+'_'+list_roadnet[1]),
        }

        deploy_dic_exp_conf = merge(config.DIC_EXP_CONF, dic_exp_conf_extra)
        deploy_dic_agent_conf = merge(getattr(config, "DIC_{0}_AGENT_CONF".format(args.mod.upper())),dic_agent_conf_extra)
        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
        deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)

        phase_map = {'cyclic': config.DIC_PHASE_MAP_CYCLIC, 'cross': config.DIC_PHASE_MAP_CROSS, 'all': config.DIC_PHASE_MAP_ALL}
        deploy_dic_traffic_env_conf["DIC_PHASE_MAP"] = phase_map[args.phasetype]

        if args.xfer is not None:
            deploy_dic_traffic_env_conf["TRANSFER"] = True
            deploy_dic_exp_conf["NUM_ROUNDS"] = 50
            deploy_dic_exp_conf["START_ROUNDS"] = 250
            args.oneprocess = True
            print('Enabling Transfer')

        deploy_dic_traffic_env_conf["LIST_STATE_FEATURE"] = ["transform_{0}".format(args.transform)]
        deploy_dic_agent_conf["D_DENSE"] = list(int(i) for i in args.dense.split(','))
        if deploy_dic_agent_conf["N_LAYER"] < len(deploy_dic_agent_conf["D_DENSE"]):
            deploy_dic_agent_conf["N_LAYER"] = len(deploy_dic_agent_conf["D_DENSE"])
            print('Increasing N_LAYER to', deploy_dic_agent_conf["N_LAYER"])
        print('D_DENSE:', deploy_dic_agent_conf["D_DENSE"])

        if args.density > 1:
            deploy_dic_traffic_env_conf["MAX_VEHS_IN_LANE"] = args.density
            print('Enabling density by', args.density)

        rew_map = { 'xcnt': 1, 'mp': -0.25, 'stop': -0.25, 'qden': -0.25 }
        dic_rew = {}
        for r in args.rewards.split('+'):
            dic_rew[r] = rew_map[r]
        deploy_dic_traffic_env_conf["DIC_REWARD_INFO"] = dic_rew

        if args.num_phase < 4:
            for i in range(4,args.num_phase,-1):
                deploy_dic_traffic_env_conf["DIC_PHASE_MAP"].pop(i-1)
            deploy_dic_traffic_env_conf["DIC_FEATURE_DIM"]['D_TRANSFORM_LANE'] = (args.num_phase*3,)
            deploy_dic_traffic_env_conf["DIC_FEATURE_DIM"]['D_TRANSFORM_APPROACH'] = (args.num_phase,)

        if args.notest:
            deploy_dic_traffic_env_conf['NO_TEST'] = args.notest
            deploy_dic_exp_conf['NUM_ROUNDS'] += 1
            print('Skip Test till', args.notest)

        if args.singleagent:
            deploy_dic_traffic_env_conf['SINGLE_AGENT'] = deploy_dic_traffic_env_conf["NUM_AGENTS"]
            deploy_dic_traffic_env_conf['ONE_MODEL'] = True
            print('Enabling Single Agent')

        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            print('Setting CUDA_VISIBLE_DEVICES to', args.gpu)
            deploy_dic_agent_conf["GPU"] = args.gpu

        deploy_dic_traffic_env_conf["LIST_STATE_FEATURE1"] = copy.deepcopy(deploy_dic_traffic_env_conf["LIST_STATE_FEATURE"])

        # For peer teaching
        if args.teach is not None:
            deploy_dic_path["PATH_TO_TEACHER_MODEL"] = args.teach
            if args.tblind:
                deploy_dic_agent_conf["EPSILON"] = deploy_dic_agent_conf["EPSILON_DECAY"] = 1
            print('Teacher :', args.teach, ', with tblind =', args.tblind)
            if args.twho is not None:
                deploy_dic_traffic_env_conf["LIST_STATE_FEATURE2"] = ["transform_{0}".format(args.twho)]
                deploy_dic_traffic_env_conf["LIST_STATE_FEATURE"] = deploy_dic_traffic_env_conf["LIST_STATE_FEATURE1"] + deploy_dic_traffic_env_conf["LIST_STATE_FEATURE2"]
                print('LIST_STATE_FEATURES', deploy_dic_traffic_env_conf["LIST_STATE_FEATURE"], deploy_dic_traffic_env_conf["LIST_STATE_FEATURE1"], deploy_dic_traffic_env_conf["LIST_STATE_FEATURE2"])

        # Print State and Rewards
        print('LIST_STATE_FEATURE', deploy_dic_traffic_env_conf["LIST_STATE_FEATURE"])
        print('DIC_REWARD_INFO', deploy_dic_traffic_env_conf["DIC_REWARD_INFO"])
        print('PHASE', deploy_dic_traffic_env_conf["DIC_PHASE_MAP"])
        print('ARGS', args)
        if args.dryrun:
            raise "Dry Run"

        # Link model to records
        if args.resume is None:
            model_dir = os.path.join(deploy_dic_path["PATH_TO_WORK_DIRECTORY"], "model")
            if args.xfer is None:
                os.makedirs(model_dir)
            else:
                os.makedirs(deploy_dic_path["PATH_TO_WORK_DIRECTORY"])
                os.symlink(os.path.join('..', args.xfer, 'model'), model_dir)

        code_dir = os.path.join(deploy_dic_path["PATH_TO_WORK_DIRECTORY"], "code_"+time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
        os.makedirs(code_dir)
        os.system('cp code/*.py ' + code_dir)

        # Write the msg
        if args.msg is not None:
            with open(os.path.join(deploy_dic_path["PATH_TO_WORK_DIRECTORY"], "readme.txt"), 'a') as file:
                print('\n', time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())), file=file)
                print(' '.join(sys.argv[:]), file=file)
                args.msg = ''
                print(args, file=file)

        pipeline_wrapper(multi_process=not args.oneprocess,
                         dic_exp_conf=deploy_dic_exp_conf,
                         dic_agent_conf=deploy_dic_agent_conf,
                         dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                         dic_path=deploy_dic_path)

    # Mark End of Experiment
    os.system('echo '+deploy_dic_path['PATH_TO_WORK_DIRECTORY'] + ' >> ' + os.path.join("records", args.memo, 'ExpDone.txt'))

    return args.memo


if __name__ == "__main__":
    args = parse_args()
   # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    main(args)
