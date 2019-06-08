# coding: utf-8

from __future__ import division


import pickle
import sys
import copy
import types
from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy
import numpy as np
import gym
from gym import spaces
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.algos.trpo import TRPO
from rllab.misc.overrides import overrides





class StudentEnv(gym.Env):

    def __init__(self, n_items=10, n_steps=100, discount=1., reward_func='likelihood'):

        self.curr_step = None
        self.n_steps = n_steps
        self.n_items = n_items
        self.now = 0
        self.curr_item = 0
        self.curr_outcome = None
        self.curr_delay = None
        self.discount = discount
        self.reward_func = reward_func
        self.action_space = spaces.Discrete(n_items)
        self.observation_space = spaces.Box(np.zeros(2), np.array([n_items - 1, 1]))

    def _recall_likelihoods(self):
        raise NotImplementedError

    def _recall_log_likelihoods(self, eps=1e-9):
        return np.log(eps + self._recall_likelihoods())

    def predict(self, q):
        raise NotImplementedError

    def _update_model(self, curr_item, curr_outcome):
        raise NotImplementedError

    def _obs(self):
        return np.array([self.curr_item, self.curr_outcome], dtype=int)

    def _rew(self):
        if self.reward_func == 'likelihood':
            return self._recall_likelihoods().mean()
        elif self.reward_func == 'log_likelihood':
            return self._recall_log_likelihoods().mean()
        else:
            raise ValueError

    def step(self, action):
        if self.curr_step is None or self.curr_step >= self.n_steps:
            raise ValueError

        if action < 0 or action >= self.n_items:
            raise ValueError

        # student model do the exercise and update model
        self.curr_item = action
        self.curr_outcome = 1 if np.random.random() < self.predict(arms[action]) else 0

        self._update_model(arms[self.curr_item], self.curr_outcome)
        self.curr_step += 1

        # if the exercise which student used to answer correctly, the reward is 0
        if self.curr_item in self.right:
            r = 0
        else:
            r = self._rew()
        if self.curr_outcome == 1 and self.curr_item not in self.right:
            self.right.append(action)

        obs = self._obs()
        done = self.curr_step == self.n_steps
        info = {}

        return obs, r, done, info

    def actualStep(self, action, answer):
        self.curr_item = action
        self.curr_outcome = answer
        self._update_model(arms[self.curr_item], self.curr_outcome)
        obs = self._obs()
        return obs

    def reset(self):
        self.curr_step = 0
        self.now = 0
        return self.step(np.random.choice(range(self.n_items)))[0]

    def recomreset(self):
        self.curr_step = 0
        self.now = 0


class DKVEnv(StudentEnv):
    def __init__(self, **kwargs):

        super(DKVEnv, self).__init__(**kwargs)

        self._init_params()

    def _init_params(self):
        """
        Init DKVMN-CA student model

        """

        # the parameters of trained DKVMN-CA model
        with open('./shulun_param.pkl', 'rb') as f:
            params = pickle.load(f)[0]

        # Knowledge Concepts Corresponding to the exercise
        with open('q2kg.pkl', 'rb') as f:
            self.q2kg = pickle.load(f)

        # contains the exercise which has already been answered correctly
        self.right = []

        self.q_embed_mtx = params['Embedding/q_embed:0']

        self.qa_embed_mtx = params['Embedding/qa_embed:0']

        self.key_matrix = params['Memory/key:0']

        self.value_matrix = params['Memory/value:0']

        self.summary_w = params['Summary_Vector/weight:0']

        self.summary_b = params['Summary_Vector/bias:0']

        self.predict_w = params['Prediction/weight:0']

        self.predict_b = params['Prediction/bias:0']

        self.erase_w = params['DKVMN_value_matrix/Erase_Vector/weight:0']

        self.erase_b = params['DKVMN_value_matrix/Erase_Vector/bias:0']

        self.add_w = params['DKVMN_value_matrix/Add_Vector/weight:0']

        self.add_b = params['DKVMN_value_matrix/Add_Vector/bias:0']

    def softmax(self, num):
        return np.exp(num) / np.sum(np.exp(num), axis=0)

    def cor_weight(self, embedded, q):
        """
        Calculate the KCW of the exercise
        :param embedded: the embedding of exercise q
        :param q: exercise ID
        :return: the KCW of the exercise
        """
        kg = self.q2kg[q]
        corr = self.softmax([np.dot(embedded, self.key_matrix[i]) for i in kg])
        correlation = np.zeros(Concepts)
        for j in range(3):
            correlation[kg[j]] = corr[j]
        return correlation


    def read(self, value_matrix, correlation_weight):
        """
        Calculate master level of concepts related to the exercise
        :param value_matrix: master level of different knowledge concepts
        :param correlation_weight: KCW of exercise
        :return: master level of concepts related to the exercise
        """
        read_content = []
        for dim in range(value_matrix.shape[0]):
            r = np.multiply(correlation_weight[dim], value_matrix[dim])
            read_content.append(r)
        read_content = np.sum(np.array(read_content), axis=0)
        return read_content

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def linear_op(self, w, b, x):
        return np.matmul(x, w) + b

    def predict(self, q):
        """
        Probability of answer exercise q correctly
        :param q: Exercise ID
        :return:Probability
        """
        cor = self.cor_weight(self.q_embed_mtx[q], q)
        read_content = self.read(self.value_matrix, cor)
        mastery_level_prior_difficulty = np.append(read_content, self.q_embed_mtx[q])
        summary_vector = np.tanh(
            self.linear_op(self.summary_w, self.summary_b, mastery_level_prior_difficulty)
        )
        pred_logits = self.linear_op(self.predict_w, self.predict_b, summary_vector)
        return self.sigmoid(pred_logits)

    def write(self, correlation_weight, qa_embed):
        """
        Update the Value_matrix
        :param correlation_weight: KCW of exercise
        :param qa_embed: the embedding of answering result
        :return: new Value_matrix
        """
        erase_vector = self.linear_op(self.erase_w, self.erase_b, qa_embed)
        erase_signal = self.sigmoid(erase_vector)
        add_vector = self.linear_op(self.add_w, self.add_b, qa_embed)
        add_signal = np.tanh(add_vector)
        for dim in range(self.value_matrix.shape[0]):
            self.value_matrix[dim] = self.value_matrix[dim] * (1 - correlation_weight[dim] * erase_signal) + \
                                     correlation_weight[
                                         dim] * add_signal
        return self.value_matrix

    def _recall_likelihoods(self):
        """
        The average probability of doing all the test exercises correctly
        """
        return np.array(list(map(self.predict, test)))


    def _update_model(self, item, outcome):
        """
        Update student model
        :param item: action(recommended exercise)
        :param outcome: answer result
        """
        ans = NumQ * outcome + item
        cor = self.cor_weight(self.q_embed_mtx[item], item)
        self.value_matrix = self.write(cor, self.qa_embed_mtx[ans])


    def reset(self):
        """
        Reset for training agent
        :return:
        """
        self._init_params()
        return super(DKVEnv, self).reset()

    def recomreset(self):
        """
        Reset for recommendation(for example, agent recommend to a new student)
        :return:
        """
        self._init_params()
        return super(DKVEnv, self).recomreset()

class Tutor(object):

    def __init__(self):
        pass

    def _next_item(self):
        raise NotImplementedError

    def _update(self, item, outcome, timestamp, delay):
        raise NotImplementedError

    def act(self, obs):
        self._update(*list(obs))
        return self._next_item()

    def learn(self, r):
        pass

    def train(self, env, n_eps=10):
        return run_eps(self, env, n_eps=n_eps)

    def reset(self):
        raise NotImplementedError


def make_rl_student_env(env):
    env = copy.deepcopy(env)

    env.n_item_feats = int(np.log(2 * env.n_items))

    env.item_feats = np.random.normal(
        np.zeros(2 * env.n_items * env.n_item_feats),
        np.ones(2 * env.n_items * env.n_item_feats)).reshape((2 * env.n_items, env.n_item_feats))

    env.observation_space = spaces.Box(
        np.concatenate((np.ones(env.n_item_feats) * -sys.maxsize, np.zeros(1))),
        np.concatenate((np.ones(env.n_item_feats) * sys.maxsize, np.ones(1)))
    )

    def encode_item(self, item, outcome):
        return self.item_feats[self.n_items * outcome + item, :]

    def vectorize_obs(self, item, outcome):
        return np.concatenate((self.encode_item(item, outcome), np.array([outcome])))
        # return self.encode_item(item, outcome)

    env._obs_orig = env._obs

    def _obs(self):
        item, outcome = env._obs_orig()

        return self.vectorize_obs(item, outcome)

    env.encode_item = types.MethodType(encode_item, env)
    env.vectorize_obs = types.MethodType(vectorize_obs, env)
    env._obs = types.MethodType(_obs, env)

    return env


from rllab.envs.gym_env import *


class MyGymEnv(GymEnv):
    def __init__(self, env, record_video=False, video_schedule=None, log_dir=None, record_log=False,
                 force_reset=False):
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
        Serializable.quick_init(self, locals())

        self.env = env
        self.env_id = ''

        assert not (not record_log and record_video)

        if log_dir is None or record_log is False:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            self.env = gym.wrappers.Monitor(self.env, log_dir, video_callable=video_schedule, force=True)
            self.monitoring = True

        self._observation_space = convert_gym_space(env.observation_space)
        logger.log("observation space: {}".format(self._observation_space))
        self._action_space = convert_gym_space(env.action_space)
        logger.log("action space: {}".format(self._action_space))
        self._horizon = self.env.n_steps
        self._log_dir = log_dir
        self._force_reset = force_reset


class DummyTutor(Tutor):

    def __init__(self, policy):
        self.policy = policy

    def act(self, obs):
        return self.policy(obs)

    def reset(self):
        pass


class LoggedTRPO(TRPO):

    def __init__(self, *args, **kwargs):
        super(LoggedTRPO, self).__init__(*args, **kwargs)
        self.rew_chkpts = []

    @overrides
    def train(self):
        self.start_worker()
        self.init_opt()
        for itr in range(self.current_itr, self.n_itr):
            paths = self.sampler.obtain_samples(itr)
            samples_data = self.sampler.process_samples(itr, paths)
            self.optimize_policy(itr, samples_data)
            my_policy = lambda obs: self.policy.get_action(obs)[0]
            r, _ = run_ep(DummyTutor(my_policy), self.env)
            self.rew_chkpts.append(r)
            print(self.rew_chkpts[-1])
        self.shutdown_worker()


class RLTutor(Tutor):

    def __init__(self, n_items, init_timestamp=0):
        self.raw_policy = None
        self.curr_obs = None
        self.rl_env = MyGymEnv(make_rl_student_env(env))

    def train(self, gym_env, n_eps=10):
        env = MyGymEnv(gym_env)
        policy = CategoricalGRUPolicy(
            env_spec=env.spec, hidden_dim=32,
            state_include_action=False)
        self.raw_policy = LoggedTRPO(
            env=env,
            policy=policy,
            baseline=LinearFeatureBaseline(env_spec=env.spec),
            batch_size=4000,
            max_path_length=env.env.n_steps,
            n_itr=n_eps,
            discount=0.99,
            step_size=0.01,
            verbose=False
        )
        self.raw_policy.train()
        return self.raw_policy.rew_chkpts

    def guide(self, obs):
        agents = DummyTutor(lambda obs: self.raw_policy.policy.get_action(obs)[0])
        action = agents.act(obs)
        return action

    def getObs(self, action, answer):
        obs = self.raw_policy.env.env.actualStep(action, answer)
        return obs

    def reset(self):
        self.curr_obs = None
        self.raw_policy.reset()

    def _next_item(self):
        if self.curr_obs is None:
            raise ValueError
        return self.raw_policy.get_action(self.curr_obs)[0]

    def _update(self, obs):
        self.curr_obs = self.vectorize_obs(obs)

    def act(self, obs):
        self._update(obs)
        return self._next_item()

    def reset(self):
        self.raw_policy.reset()


def run_ep(agent, env):
    agent.reset()
    obs = env.reset()
    done = False
    totalr = []
    observations = []
    print('run_ep')
    while not done:
        action = agent.act(obs)
        obs, r, done, _ = env.step(action)
        agent.learn(r)
        totalr.append(r)
        observations.append(obs)
    return np.mean(totalr), observations


def all_reset(agent):
    """
    Reset policy and student model for recommendation (when the agent recommends to a new student)

    """
    agent.raw_policy.env.env.recomreset()
    agent.raw_policy.policy.reset()

    return agent


def simulation(agent, trace, steps):
    """
    Simulate the recommendation given the student history exercise trace
    :param agent: recommendation policy
    :param trace: student history exercise trace
    :param steps: the number of exercises recommended to the student
    :return: recommended exercises and his predicted knowledge status
    """
    recom_trace = []
    a2i = dict(zip(arms, range(len(arms))))
    trace = [(a2i[i[0]], i[1]) for i in trace]
    for q, a in trace:
        obs = agent.raw_policy.env.env.vectorize_obs(q, a)
        recomq = agent.guide(obs)

    res = []
    for t in range(steps):
        prob = agent.raw_policy.env.env.predict(arms[recomq])
        answer = 1 if np.random.random() < prob else 0

        # obs = agent.raw_policy.env.env.vectorize_obs(recomq, answer)
        recom_trace.append((recomq, answer))
        obs = agent.raw_policy.env.env.actualStep(recomq, answer)
        res.append(np.mean(list(map(agent.raw_policy.env.env.predict, arms))))
        recomq = agent.guide(obs)

    return recom_trace, res


def evalution(agent):
    """
    Evaluate the policy when it recommend exercises to different student
    allshulun:[[(923, 1), (175, 0), (1010, 1), (857, 0), (447, 0)], [........], [.........]]
    :param agent:
    :return: different students'predicted knowledge status
    """
    with open('./好未来数据/allshulun.pkl', 'rb') as f:
        allshulun = pickle.load(f)
    allre = [[] for i in range(50)]
    for trace in allshulun:
        agent = all_reset(agent)
        t, res = simulation(agent, trace, 50)
        for j in range(50):
            allre[j].append(res[j])
    result = [np.mean(k) for k in allre]
    return result


def run_eps(agent, env, n_eps=100):
    tot_rew = []
    for i in range(n_eps):
        totalr, _ = run_ep(agent, env)
        tot_rew.append(totalr)
    return tot_rew

# the recommended candidate sets of exercises
with open('arms.pkl', 'rb') as f:
    arms = pickle.load(f)

test = arms
Concepts = 188
NumQ = 1982
n_steps = 30
n_items = len(arms)
discount = 0.99
n_eps = 200

reward_funcs = ['likelihood']
envs = [
    ('DKVMN', DKVEnv)
]


tutor_builders = [
    ('RL', RLTutor)
]

env_kwargs = {
    'n_items': n_items, 'n_steps': n_steps, 'discount': discount
}

env = DKVEnv(**env_kwargs, reward_func='likelihood')
rl_env = make_rl_student_env(env)
agent = RLTutor(n_items)
reward = agent.train(rl_env, n_eps=n_eps)
print('ok')