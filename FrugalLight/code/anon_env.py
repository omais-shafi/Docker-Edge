import os
import sys
import json
import time
import pickle
import itertools
import numpy as np
import pandas as pd
from multiprocessing import Process
import cityflow

print_roads = False

class Intersection:

    def __init__(self, inter_id, dic_traffic_env_conf, env):
        self.inter_id = inter_id
        self.inter_name = "intersection_{0}_{1}".format(inter_id[0], inter_id[1])

        self.env = env
        self.eng = env.eng

        self.max_vehs = dic_traffic_env_conf["MAX_VEHS_IN_LANE"] if "MAX_VEHS_IN_LANE" in dic_traffic_env_conf.keys() else 1
        self.num_lanes = sum(list(dic_traffic_env_conf["LANE_NUM"].values()))
        self.num_approaches = 4
        self.dic_phase_map = dic_traffic_env_conf["DIC_PHASE_MAP"]
        self.num_phases = len(self.dic_phase_map)-1
        if self.num_approaches > self.num_phases:
            self.num_approaches = self.num_phases
        self.DIC_PHASE_GREEN_APPROACH_MAP = dic_traffic_env_conf["DIC_PHASE_GREEN_APPROACH_MAP"]

        # =====  intersection settings, Sequencing Crucial for rotation symmetry  =====
        self.list_approachs = ["W", "S", "E", "N"]
        self.dic_approach_to_node = {"W": 0, "S": 1, "E": 2, "N": 3}
        self.dic_entering_approach_to_edge = {"W": "road_{0}_{1}_0".format(inter_id[0] - 1, inter_id[1])}
        self.dic_entering_approach_to_edge.update({"S": "road_{0}_{1}_1".format(inter_id[0], inter_id[1] - 1)})
        self.dic_entering_approach_to_edge.update({"E": "road_{0}_{1}_2".format(inter_id[0] + 1, inter_id[1])})
        self.dic_entering_approach_to_edge.update({"N": "road_{0}_{1}_3".format(inter_id[0], inter_id[1] + 1)})

        self.dic_edge_to_index = dict()
        for key,idx in self.dic_approach_to_node.items():
            self.dic_edge_to_index[ self.dic_entering_approach_to_edge[key] ] = idx

        self.dic_exiting_approach_to_edge = { approach: "road_{0}_{1}_{2}".format(inter_id[0], inter_id[1], self.dic_approach_to_node[approach]) for approach in self.list_approachs}
        if print_roads:
            print('Inter', inter_id, self.dic_entering_approach_to_edge, '|', self.dic_exiting_approach_to_edge)

        # generate all lanes
        self.list_entering_lanes = []
        for approach in self.list_approachs:
            self.list_entering_lanes += [self.dic_entering_approach_to_edge[approach] + '_' + str(i) for i in range(self.num_lanes)]
        self.list_exiting_lanes = []
        for approach in self.list_approachs:
            self.list_exiting_lanes += [self.dic_exiting_approach_to_edge[approach] + '_' + str(i) for i in range(self.num_lanes)]

        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        # previous & current
        self.dic_lane_vehicle_previous_step = {}
        self.mark_vehs_for_exit_check = []

        # -1: all yellow, -2: all red, -3: none
        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2

        self.current_phase_index = 0
        self.previous_phase_index = 0
        self.eng.set_tl_phase(self.inter_name, self.current_phase_index)

        self.next_phase_to_set_index = 1
        self.current_phase_duration = -1
        self.all_red_flag = False
        self.all_yellow_flag = True

        self.dic_feature = {}  # this second

    def set_signal(self, action, action_pattern, yellow_time, all_red_time):
        if self.all_yellow_flag:
            # in yellow phase
            if self.current_phase_duration >= yellow_time: # yellow time reached
                self.current_phase_index = self.next_phase_to_set_index
                nxt = self.dic_phase_map[self.current_phase_index]
                self.eng.set_tl_phase(self.inter_name, nxt)
                self.all_yellow_flag = False
        else:
            # determine phase
            if action_pattern == 0: # set to certain phase
                self.next_phase_to_set_index = action
            elif action <= action_pattern < self.num_phases: # switch by order
                # change to the next phase
                self.next_phase_to_set_index = self.current_phase_index + action
                if self.next_phase_to_set_index >= self.num_phases:
                    self.next_phase_to_set_index = 0
            else:
                sys.exit("action not recognized\n action must be 0 or 1")

            # set phase
            if self.current_phase_index != self.next_phase_to_set_index:
                # the light phase needs to change
                # change to yellow first, and activate the counter and flag
                self.eng.set_tl_phase(self.inter_name, 0)
                # Let it retain the existing phase during yellow light
                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True

    def find_veh_lane(self, veh):
        for lane in self.list_entering_lanes:
            if veh in self.dic_lane_vehicle_current_step[lane]:
                return lane, self.dic_edge_to_index[lane[0:-2]]
        return '',''

    # update inner measurements
    def update_previous_measurements(self):
        self.previous_phase_index = self.current_phase_index
        self.dic_lane_vehicle_previous_step = self.dic_lane_vehicle_current_step

    def update_current_measurements_map(self, simulator_state, path_to_log,test_flag):

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        for lane in self.list_lanes:
            self.dic_lane_vehicle_current_step[lane] = simulator_state["get_lane_vehicles"][lane]
        for lane in self.list_entering_lanes:
            self.dic_lane_waiting_vehicle_count_current_step[lane] = simulator_state["get_lane_waiting_vehicle_count"][lane]
        for lane in self.list_exiting_lanes:
            self.dic_lane_waiting_vehicle_count_current_step[lane] = simulator_state["get_lane_waiting_vehicle_count"][lane]

        self._update_first_entering_approach_vehicle()
        self.num_vehicle_new_left_entering_lane = len(self._update_leave_entering_approach_vehicle())

        # update feature
        self._update_feature_map()

    def _update_first_entering_approach_vehicle(self):
        ts = self.get_current_time()
        for lane in self.list_entering_lanes:
            if lane[0:-2] in self.env.list_entering_approaches:
                for veh in self.dic_lane_vehicle_current_step[lane]:
                    if veh not in self.env.vehs_enter_exit.keys():
                        self.env.vehs_enter_exit[veh] = {"enter_time": ts, "prev_time": ts, "stuck_time": 0, "leave_time": 0, "enter_inter": "{0}_{1}".format(self.inter_id[1]-1,self.dic_edge_to_index[lane[0:-2]]), "inter_xed": 0}

    def _update_leave_entering_approach_vehicle(self):

        if not self.dic_lane_vehicle_previous_step:
            return []

        ts = self.get_current_time()
        list_entering_lane_vehicle_left = []
        last_step_vehicle_id_list = []
        current_step_vehilce_id_list = []
        for lane in self.list_entering_lanes:
            last_step_vehicle_id_list.extend(self.dic_lane_vehicle_previous_step[lane])
            current_step_vehilce_id_list.extend(self.dic_lane_vehicle_current_step[lane])

        list_entering_lane_vehicle_left.extend(
            list(set(last_step_vehicle_id_list) - set(current_step_vehilce_id_list))
        )

        # Manage global vehicle stats
        for idx, veh in enumerate(self.mark_vehs_for_exit_check):
            ret = self._check_veh_exiting_from_network(veh)
            if ret != 0:
                self.mark_vehs_for_exit_check.pop(idx)

        for veh in list_entering_lane_vehicle_left:
            obj = self.env.vehs_enter_exit[veh]
            obj["inter_xed"] += 1
            obj["leave_time"] = ts
            stuck_dur = ts - obj["prev_time"]
            if stuck_dur > obj["stuck_time"]:
                obj["stuck_time"] = stuck_dur
            obj["prev_time"] = ts

            ret = self._check_veh_exiting_from_network(veh)
            if ret == 0:
                # Vehicle is in between intersection, wait for it to land an approach
                self.mark_vehs_for_exit_check.append(veh)

        return list_entering_lane_vehicle_left

    def _check_veh_exiting_from_network(self, veh):
        for lane in self.list_exiting_lanes:
            if lane[0:-2] in self.env.list_exiting_approaches:
                if veh in self.dic_lane_vehicle_current_step[lane]:
                    return 1
            if lane[0:-2] in self.env.list_internal_approaches:
                if veh in self.dic_lane_vehicle_current_step[lane]:
                    self.env.vehs_enter_exit[veh]["leave_time"] = 0
                    return -1
        return 0

    def transform_lane(self, wsen_vehs):
        map = self.DIC_PHASE_GREEN_APPROACH_MAP[self.dic_phase_map[self.current_phase_index]]
        if not len(map):
            return [0 for i in range(self.num_approaches*self.num_lanes)]
        rotate = map[0]
        wsen_vehs_r = wsen_vehs[rotate*self.num_lanes:self.num_approaches*self.num_lanes]
        wsen_vehs_r.extend(wsen_vehs[0:rotate*self.num_lanes])
        if self.max_vehs > 1:
            wsen_vehs_r = [min(x/self.max_vehs, 1.0) for x in wsen_vehs_r]
        return wsen_vehs_r

    def transform_approach(self, wsen_vehs):
        wsen_approach = []
        for i in range(self.num_approaches):
            wsen_approach.append(0)
            for j in range(self.num_lanes):
                wsen_approach[-1] += wsen_vehs[i*self.num_lanes+j]
            wsen_approach[-1] /= self.num_lanes
        return wsen_approach

    def transform_group(self, wsen_approach):
        map = self.DIC_PHASE_GREEN_APPROACH_MAP[self.dic_phase_map[self.current_phase_index]]
        if len(map) == 0:
            return [0, np.sum(wsen_approach)]
        if len(map) == 1:
            return [wsen_approach[0], np.sum(wsen_approach[1:self.num_approaches])]
        return [wsen_approach[0]+wsen_approach[2], wsen_approach[1]+wsen_approach[3]]

    def transform_relative(self, wsen_group):
        den = np.sum(wsen_group)
        if den < 1/self.max_vehs:
            return [0]
        return [wsen_group[0] / den]

    def _update_feature_map(self):
        dic_feature = dict()

        dic_feature["transform_lane"] = self.transform_lane(self._get_lane_num_vehicle(self.list_entering_lanes))
        dic_feature["transform_approach"] = self.transform_approach(dic_feature["transform_lane"])
        dic_feature["transform_group"] = self.transform_group(dic_feature["transform_approach"])
        dic_feature["transform_relative"] = self.transform_relative(dic_feature["transform_group"])

        dic_feature["num_vehicle_left"] = [self.num_vehicle_new_left_entering_lane]
        dic_feature["pressure"] = self._get_pressure()
        dic_feature["vehicles_been_stopped_thres1"] = self._get_lane_num_vehicle_been_stopped(1,self.list_entering_lanes)

        self.dic_feature = dic_feature

    def _get_pressure(self):
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_entering_lanes] + \
        [-self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_exiting_lanes]

    def _get_lane_num_vehicle(self, list_lanes):
        ''' vehicle number for each lane '''
        return [len(self.dic_lane_vehicle_current_step[lane]) for lane in list_lanes]

    def _get_lane_num_vehicle_been_stopped(self, thres, list_lanes):
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in list_lanes]

    # ================= get functions from outside ======================
    def get_current_time(self):
        return self.eng.get_current_time()

    def get_feature(self):
        return self.dic_feature

    def get_state(self, list_state_features):
        return {state_feature_name: self.dic_feature[state_feature_name] for state_feature_name in list_state_features}


class AnonEnv:
    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf):
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None
        self.system_states = None

        # check min action time
        if self.dic_traffic_env_conf["MIN_ACTION_TIME"] <= self.dic_traffic_env_conf["YELLOW_TIME"]:
            print ("MIN_ACTION_TIME should include YELLOW_TIME")
            raise ValueError

        # touch new inter_{}.pkl (if exists, remove)
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            f.close()

    def reset(self):

        cityflow_config = {
            "interval": self.dic_traffic_env_conf["INTERVAL"],
            "seed": 0,
            "laneChange": False,
            "dir": self.path_to_work_directory+"/",
            "roadnetFile": self.dic_traffic_env_conf["ROADNET_FILE"],
            "flowFile": self.dic_traffic_env_conf["TRAFFIC_FILE"],
            "rlTrafficLight": self.dic_traffic_env_conf["RLTRAFFICLIGHT"],
            "saveReplay": self.dic_traffic_env_conf["SAVEREPLAY"],
            "roadnetLogFile": "frontend/web/roadnetLogFile.json",
            "replayLogFile": "frontend/web/replayLogFile.txt"
        }
        print("=========================")
        print(cityflow_config)

        with open(os.path.join(self.path_to_work_directory,"cityflow.config"), "w") as json_file:
            json.dump(cityflow_config, json_file)
        self.eng = cityflow.Engine(os.path.join(self.path_to_work_directory,"cityflow.config"), thread_num=1)

        # initialize intersections (grid)
        self.list_intersection = [Intersection((i+1, j+1), self.dic_traffic_env_conf, self)
                                  for i in range(self.dic_traffic_env_conf["NUM_ROW"])
                                  for j in range(self.dic_traffic_env_conf["NUM_COL"])]
        self.list_inter_log = [[] for _ in range(self.dic_traffic_env_conf["NUM_ROW"] *
                                                 self.dic_traffic_env_conf["NUM_COL"])]

        self.id_to_index = {}
        count = 0
        for i in range(self.dic_traffic_env_conf["NUM_ROW"]):
            for j in range(self.dic_traffic_env_conf["NUM_COL"]):
                self.id_to_index['intersection_{0}_{1}'.format(i+1, j+1)] = count
                count += 1

        self.list_lanes = []
        for inter in self.list_intersection:
            self.list_lanes += inter.list_lanes
        self.list_lanes = np.unique(self.list_lanes).tolist()

        # Find final exiting approaches
        list_enter = []
        list_exit = []
        for inter in self.list_intersection:
            list_enter += list(inter.dic_entering_approach_to_edge.values())
            list_exit  += list(inter.dic_exiting_approach_to_edge.values())
        list_enter = np.unique(list_enter).tolist()
        list_exit = np.unique(list_exit).tolist()

        self.vehs_enter_exit = dict()
        self.list_entering_approaches = []
        self.list_exiting_approaches = []
        self.list_internal_approaches = []
        for en in list_enter:
            if en not in list_exit:
                self.list_entering_approaches.append(en)
        for ex in list_exit:
            if ex not in list_enter:
                self.list_exiting_approaches.append(ex)
        for ex in list_exit:
            if ex in list_enter:
                self.list_internal_approaches.append(ex)
        if print_roads:
            print('Entering', len(self.list_entering_approaches), self.list_entering_approaches)
            print('Exiting', len(self.list_exiting_approaches), self.list_exiting_approaches)
            print('Internal', len(self.list_internal_approaches), self.list_internal_approaches)
            print('Lanes', len(self.list_lanes), self.list_lanes)

        self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance(),
                              "get_vehicle_speed": None
                              }

        for inter in self.list_intersection:
            inter.update_current_measurements_map(self.system_states,self.path_to_log,False)

        state, done = self.get_state()
        return state

    def action_control(self, action):
        return [action[i][0] for i in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"])]

    def step(self, action, test_flag):

        action = self.action_control(action)

        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]-1):
            if self.dic_traffic_env_conf["ACTION_PATTERN"]:
                list_action_in_sec.append(np.zeros_like(action).tolist())
            else: #if self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(np.full_like(action, fill_value=-1).tolist())

        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]):

            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]

            instant_time = self.get_current_time()
            self.current_time = self.get_current_time()
            before_action_feature = self.get_feature()
            self._inner_step(action_in_sec,test_flag)
            self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action_in_sec_display)
            next_state, done = self.get_state()

        return next_state, None, done

    def _inner_step(self, action,test_flag):

        # copy current measurements to previous measurements
        for inter in self.list_intersection:
            inter.update_previous_measurements()

        # set signals
        # multi_intersection decided by action {inter_id: phase}
        for inter_ind, inter in enumerate(self.list_intersection):
            inter.set_signal(
                action=action[inter_ind],
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
                all_red_time=self.dic_traffic_env_conf["ALL_RED_TIME"]
            )

        # run one step
        for i in range(int(1/self.dic_traffic_env_conf["INTERVAL"])):
            self.eng.next_step()

        self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance(),
                              "get_vehicle_speed": None
                              }

        for inter in self.list_intersection:
            inter.update_current_measurements_map(self.system_states,self.path_to_log,test_flag)

    def log_metrics(self):

        ts = self.get_current_time()
        num_vehs = len(self.vehs_enter_exit.keys())
        num_exit = 0
        total_time = 0.0
        total_time2 = 0.0
        with open(os.path.join(self.path_to_log, "metrics.csv"), 'w') as file:
            print('ID,Enter,"=COUNTIF(C2:C{0},">0")",=SUM(D2:D{0})/C1,=SUM(E2:E{0})/({0}-1-C1),Wait,Inter,xed'.format(num_vehs+1), file=file)
            for vehicle in self.vehs_enter_exit.keys():
                obj = self.vehs_enter_exit[vehicle]
                enter_time = obj["enter_time"]
                leave_time = obj["leave_time"]
                if leave_time > 0:
                    travel_time = leave_time - enter_time
                    travel_time2 = 0
                    num_exit += 1
                else:
                    travel_time = 0
                    travel_time2 = ts - enter_time
                    stuck_dur = ts - obj["prev_time"]
                    if obj["stuck_time"] < stuck_dur:
                        obj["stuck_time"] = stuck_dur

                total_time += travel_time
                total_time2 += travel_time2
                print('{0},{1},{2},{3},{4},{5},{6},{7}'.format(vehicle, enter_time, leave_time, travel_time, travel_time2,
                        obj["stuck_time"], obj["enter_inter"], obj["inter_xed"]), file=file)

        if num_exit == 0:
            num_exit = -1
        # Log network level vehicle metrics
        file_name = os.path.join(self.path_to_work_directory, "metrics.csv")
        if not os.path.exists(file_name):
            with open(file_name, 'w') as file:
                print("InVeh,OutVeh,%Thru,TravelTime,StuckTime,TotalTime", sep=',', file=file)
        with open(file_name, 'a') as file:
            print("{0},{1},{2:.2f},{3:.2f},{4:.2f},{5:.2f}".format(num_vehs, num_exit, num_exit*100.0/num_vehs, total_time/num_exit, total_time2/(num_vehs-num_exit), (total_time+total_time2)/num_vehs),
                  sep=',', file=file)

    def get_feature(self):
        list_feature = [inter.get_feature() for inter in self.list_intersection]
        return list_feature

    def _check_episode_done(self, list_state):
        # ======== to implement ========
        return False

    def get_state(self):
        list_state = [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"]) for inter in self.list_intersection]
        done = self._check_episode_done(list_state)
        return list_state, done

    def get_current_time(self):
        return self.eng.get_current_time()

    def log(self, cur_time, before_action_feature, action):
        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append({"time": cur_time,
                                                    "state": before_action_feature[inter_ind],
                                                    "action": action[inter_ind]})

    def batch_log(self, start, stop):
        for inter_ind in range(start, stop):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()

    def bulk_log_multi_process(self, batch_size=100):
        assert len(self.list_intersection) == len(self.list_inter_log)
        if batch_size > len(self.list_intersection):
            batch_size_run = len(self.list_intersection)
        else:
            batch_size_run = batch_size
        process_list = []
        for batch in range(0, len(self.list_intersection), batch_size_run):
            start = batch
            stop = min(batch + batch_size, len(self.list_intersection))
            p = Process(target=self.batch_log, args=(start,stop))
            print("before")
            p.start()
            print("end")
            process_list.append(p)
        print("before join")

        for t in process_list:
            t.join()

        f = open(os.path.join(self.path_to_log, "log_done.txt"), "a")
        f.close()

    def end_anon(self):
        print("anon process end")
        pass
