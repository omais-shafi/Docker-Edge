from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers.merge import concatenate
from agent import Agent

class DQN(Agent):

    def build_network(self):

        '''Initialize a Q network'''
        LIST_STATE_FEATURE = "LIST_STATE_FEATURE1"

        # initialize feature node
        dic_input_node = {}
        for feature_name in self.dic_traffic_env_conf[LIST_STATE_FEATURE]:
            if "transform" in feature_name:
                _shape = self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()]
            else:
                _shape = (self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()][0]*self.num_lanes,)
            dic_input_node[feature_name] = Input(shape=_shape, name="input_"+feature_name)

        # add cnn to image features
        list_all_flatten_feature = []
        for feature_name in self.dic_traffic_env_conf[LIST_STATE_FEATURE]:
            if len(self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()]) > 1:
                list_all_flatten_feature.append(Flatten()(dic_input_node[feature_name]))
            else:
                list_all_flatten_feature.append(dic_input_node[feature_name])

        # concatenate features
        if len(list_all_flatten_feature) > 1:
            all_flatten_feature = concatenate(list_all_flatten_feature, axis=1, name="all_flatten_feature")
        else:
            all_flatten_feature = list_all_flatten_feature[0]

        # shared dense layer, N_LAYER
        locals()["dense_0"] = Dense(self.dic_agent_conf["D_DENSE"][0], activation="relu", name="dense_0")(all_flatten_feature)
        for i in range(1, self.dic_agent_conf["N_LAYER"]):
            idx = i if i<len(self.dic_agent_conf["D_DENSE"]) else -1
            locals()["dense_%d" % i] = Dense(self.dic_agent_conf["D_DENSE"][idx], activation="relu",
                                             name="dense_%d" % i)(locals()["dense_%d" % (i - 1)])

        q_values = Dense(self.num_actions, activation="linear", name="q_values")(locals()["dense_%d" % (self.dic_agent_conf["N_LAYER"] - 1)])

        network = Model(inputs=[dic_input_node[feature_name] for feature_name in self.dic_traffic_env_conf[LIST_STATE_FEATURE]],
                        outputs=q_values)

        network.compile(optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])

        if self.intersection_id == '0':
            network.summary()

        return network
