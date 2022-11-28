import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from functools import reduce
from utils.convert2base import obs_to_int_pi,int_to_obs,s_to_sp
import sys
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)  # control the display precision of decimals in Python
from env.env_factory import get_env
import os
import math
import argparse
import random
import collections
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--env', type=str, default='room', help='room|room30|secret_room|push_box|island')
    parser.add_argument('--no_range_info', default=False, action='store_true')
    parser.add_argument('--mixed_explore', default=False, action='store_true')
    parser.add_argument('--random_sample', default=False, action='store_true')
    parser.add_argument('--total_episode', type=int, default=50000)
    parser.add_argument('--eval_every_episode', type=int, default=200)
    parser.add_argument('--log_folder', type=str, default='.')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--alpha1', type=float, default=0.1, help='learning rate for exploration policy')
    parser.add_argument('--alpha2', type=float, default=0.05, help='learning rate for active policy')
    parser.add_argument('--recip_t', type=float, default=50, help='reciprocal temperature')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--exp_batch_name', type=str, default='exp_batch_no_name')
    parser.add_argument('--replay_size', type=int, default=400000)
    parser.add_argument('--bonus_coef', type=float, default=0.05)
    args = parser.parse_args()
    return args

args = get_args()
log_path = 'log/{}/{}/{}_{}'.format(args.log_folder, args.exp_batch_name, args.exp_name, args.seed)
os.makedirs(log_path, exist_ok=True)
seed = args.seed
np.random.seed(seed)
random.seed(seed)
env = get_env(args.env)()
eval_env = get_env(args.env)()
base = env.grid_size
raw_obs_dim = env.observation_space.nvec.size
N = 100  # how many epochs to expand the dimension of Projection matrix

def eval(policy1, policy2, eval_env, raw_obs_dim):
    episode_rew = 0
    s = eval_env.reset()
    episode_rews = []
    for _ in range(10):
        while True:
            s = obs_to_int_pi(s, base=eval_env.grid_size, raw_dim=raw_obs_dim)
            a1 = policy1.select_action(s, 0)
            a2 = policy2.select_action(s, 0)
            s_next, r, done = eval_env.step([a1, a2])
            episode_rew += r
            s = s_next
            if done:
                episode_rews.append(episode_rew)
                episode_rew = 0
                s = eval_env.reset()
                break
    return np.mean(episode_rews)

class Storage(object):
    def __init__(self, max_size=50000):
        self.state = np.zeros(max_size, dtype=np.int32)
        self.next_state = np.zeros(max_size, dtype=np.int32)
        self.action = np.zeros(max_size, dtype=np.int32)
        self.reward = np.zeros(max_size)
        self.done = np.zeros(max_size)     # 决定是否将环境初始化(即是否结束循环/迭代)
        self.time_stamp = np.zeros(max_size, dtype=np.int32)
        self.size = 0
        self.iterator = 0
        self.t = 0
        self.max_size = max_size

    def insert(self, s, a, r, ns, done):
        self.time_stamp[self.iterator] = self.t
        self.state[self.iterator] = s
        self.next_state[self.iterator] = ns
        self.action[self.iterator] = a
        self.reward[self.iterator] = r
        self.done[self.iterator] = done
        self.size += 1
        self.t += 1
        self.iterator += 1
        self.iterator = self.iterator % self.max_size     # 取余操作：圆形结构，循环分配？？？
        self.size = min(self.max_size, self.size)  # 数据集中数据量不能超过定义的最大值
        self.t = 0 if done else self.t

    def get_all_data(self):
        inds = reversed(range(self.size))
        for i in inds:
            yield self.state[i], self.action[i], self.reward[i], self.next_state[i], self.done[i]   # 生成器

    def get_prioritized_batch(self, bz=512): # 512 for pass/secret/push
        data_inds = []    # 空列表
        # For DEBUG
        pos_rew_inds = np.where(self.reward >= 1)
        if pos_rew_inds[0].size == 0:
            return
        for ind in pos_rew_inds:
            ind = ind[0]
            t = self.time_stamp[ind]
            data_inds.extend(list(range(ind - t, ind + 1)))
        data_inds.reverse()
        random_inds = random.choices(list(range(self.size)), k=bz - len(data_inds))
        data_inds.extend(random_inds)
        for i in data_inds:
            yield self.state[i], self.action[i], self.reward[i], self.next_state[i], self.done[i]

    def get_random_batch(self, bz=512): # 512 for pass/secret/push
        random_inds = random.choices(list(range(self.size)), k=bz)
        for i in random_inds:
            yield self.state[i], self.action[i], self.reward[i], self.next_state[i], self.done[i]

    def reset(self):
        self.state[:] = 0
        self.next_state[:] = 0
        self.action[:] = 0
        self.reward[:] = 0
        self.done[:] = 0
        self.time_stamp[:] = 0
        self.size = 0
        self.t = 0
        self.iterator = 0

    def batch_insert(self, s, a, r, ns, done):
        """
        Batch insert data. Don't keep track of time stamp. Assume storage will not be full.
        Should only be used for goal_replay_buffer
        """
        bz = len(s)
        if self.iterator + bz < self.max_size:    # 有足够空余允许全部插入
            self.state[self.iterator : self.iterator + bz] = s
            self.next_state[self.iterator : self.iterator + bz] = ns
            self.action[self.iterator : self.iterator + bz] = a
            self.reward[self.iterator : self.iterator + bz] = r
            self.done[self.iterator : self.iterator + bz] = done
            self.iterator += bz
            self.size += bz
        else:
            right_len = self.max_size - self.iterator
            left_len = bz - right_len
            self.state[self.iterator: self.iterator + right_len] = s[: right_len]
            self.next_state[self.iterator: self.iterator + right_len] = ns[: right_len]
            self.action[self.iterator: self.iterator + right_len] = a[: right_len]
            self.reward[self.iterator: self.iterator + right_len] = r[: right_len]
            self.done[self.iterator: self.iterator + right_len] = done[: right_len]

            self.state[:left_len] = s[right_len:]
            self.next_state[:left_len] = ns[right_len:]
            self.action[:left_len] = a[right_len:]
            self.reward[:left_len] = r[right_len:]
            self.done[:left_len] = done[right_len:]
            self.size = self.max_size
            self.iterator = left_len

    def get_goal_tractories(self, goal: (int, int), goal_replay_buffer: 'Storage', base: int):
        goal_s, goal_a = goal
        goal_s_inds = np.where(self.state == goal_s)
        if goal_s_inds[0].size == 0:
            return False
        if goal_s_inds[0].size == 0:
            goal_s = s_to_sp(goal_s, base=base,raw_dim=raw_obs_dim)
            goal_s_inds = np.where(self.state == goal_s)
        for ind in goal_s_inds:
            ind = ind[0]
            t = self.time_stamp[ind]
            goal_replay_buffer.batch_insert(self.state[ind - t: ind], self.action[ind - t: ind],
                                            self.reward[ind - t: ind], self.next_state[ind - t: ind],
                                            self.done[ind - t: ind])
            goal_replay_buffer.insert(goal_s, goal_a, 1, goal_s, 1)
        return True

    def sample_states(self, bz):
       inds = np.random.choice(range(self.size), size=min(bz, self.size), replace=False)
       return self.state[inds], inds

class QLearning(object):
    def __init__(self, n_states, n_actions, base, raw_dim, observation_space=None, gamma=0.9, alpha=0.2):
        self.q_table = np.zeros((n_states, n_actions))
        self.gamma = gamma
        self.alpha = alpha
        self.n_actions = n_actions
        self.count = np.zeros((n_states, n_actions))
        nvec = observation_space.nvec
        self.n_vars = len(observation_space.nvec)
        self.nvec = nvec
        self.base, self.raw_dim = base, raw_dim

    def update_q(self, s, a, r, s_next, done):
        if not done:
            self.q_table[s, a] = (1 - self.alpha) * self.q_table[s, a] \
                             + self.alpha * (r + self.gamma * np.max(self.q_table[s_next]))
        else:
            self.q_table[s, a] = (1 - self.alpha) * self.q_table[s, a] + self.alpha * r

    def select_action(self, s, eps, other_q_table=None, alpha=None):
        if np.random.rand() < eps:
            return np.random.choice(self.n_actions)
        else:
            # break tie uniformly
            if other_q_table is None:
                return np.random.choice(np.flatnonzero(self.q_table[s] == self.q_table[s].max()))
            else:
                q_table = (1 - alpha) * self.q_table[s] + alpha * other_q_table[s]
                return np.random.choice(np.flatnonzero(q_table == q_table.max()))

    def update_count(self, s, a):
        self.count[s, a] += 1


class PERLQLearning(QLearning):
    def __init__(self, n_states, n_actions, base, raw_dim, observation_space=None, gamma=0.9, alpha=0.2, goal_q_len=300,
                   recip_t=50, replay_size=50000, priority_sample=True):
        super(PERLQLearning, self).__init__(n_states, n_actions, base, raw_dim, observation_space, gamma, alpha)
        self.replay_buffer = Storage(replay_size)
        self.goal_replay_buffer = Storage(replay_size)  # store trajectories that attended to the goal
        self.base = base
        self.goal_q = collections.deque(maxlen=goal_q_len)
        self.n_updates = 0
        self.compute_ent_every = 20
        self.recip_t = recip_t
        self.priority_sample = priority_sample

    def insert_data(self, s, a, r, s_next, done, th=10000):
        if self.count[s, a] < th:
            self.replay_buffer.insert(s, a, r, s_next, done)

    def get_P(self,i, data):
        '''
        :param i: The row dimension of p determined by episode_count
        :param data: The augmented matrix consist of corresponding state-action
        :return: a particle matrix related to episode_count
        '''

        tf.compat.v1.reset_default_graph()
        rm=random.randint(0,512)
        X = tf.compat.v1.placeholder(tf.float32, [raw_obs_dim+2, None])
        P = tf.compat.v1.Variable(tf.compat.v1.random_normal([i, raw_obs_dim+2]))
        A = tf.matmul(P, X)
        A1 = tf.multiply(A, A)
        rdc_sum = tf.reduce_sum(A1, 1)
        lg = tf.math.log(rdc_sum)
        lg_sum = tf.reduce_sum(lg)
        cost = tf.reduce_mean(1 - lg_sum)
        learning_rate = 0.1
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        training_epochs = 200
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for epoch in range(training_epochs):
                sess.run(optimizer, feed_dict={X: data[:,rm:rm+512]})
            p = np.mat(sess.run(P))
        return p

    def update_q_from_D(self, epoch=1, reset_q=True, goal=None):
        goal_s = goal[0] if goal is not None else -1
        goal_a = goal[1] if goal is not None else -1
        success = True

        if reset_q:
            self.reset_q()
            self.goal_replay_buffer.reset()
            success = self.replay_buffer.get_goal_tractories(goal, self.goal_replay_buffer,self.base)
        if success:
            for _ in range(epoch):
                if goal is not None:
                    data = self.goal_replay_buffer.get_all_data()
                else:
                    if self.priority_sample:
                        data = self.replay_buffer.get_prioritized_batch()
                    else:
                        data = self.replay_buffer.get_random_batch()
                for s, a, r, ns, done in data:     #
                    if goal is not None:

                        if s == goal_s and a == goal_a:   # p*s里的元素类型要转为整型
                            r = 1
                        else:
                            r = 0
                    self.update_q(s, a, r, ns, done)

    def _prioritize_pos(self):
        inds = np.where(self.replay_buffer.reward > 0)    # np.where()返回的是值的位置索引
        for ind in inds:
            ind = ind[0]
            t = self.replay_buffer.time_stamp[ind]
            for _ in range(5):
                self.replay_buffer.batch_insert(self.replay_buffer.state[ind - t: ind + 1], self.replay_buffer.action[ind - t: ind + 1],
                                            self.replay_buffer.reward[ind - t: ind + 1], self.replay_buffer.next_state[ind - t: ind + 1],
                                            self.replay_buffer.done[ind - t: ind + 1])

    def reset_q(self):
        self.q_table[:] = 0

def main():

    n_states = reduce(lambda x, y: x * y, env.observation_space.nvec)
    kargs = {'n_states': n_states,
             'n_actions': reduce(lambda x, y: x * y, env.action_space.nvec),
             'base': env.grid_size,
             'raw_dim': len(env.observation_space.nvec),
             'observation_space': env.observation_space,
             'gamma': args.gamma,
             'alpha': args.alpha1,
             }
    meta_counter=PERLQLearning(**kargs)
    kargs.update({
            'recip_t': args.recip_t,
            'replay_size': args.replay_size,
            'priority_sample': not args.random_sample
        })

    meta_q = PERLQLearning(**kargs)
    kargs['n_actions'] = env.action_space.nvec[0]
    q_learner1 = PERLQLearning(**kargs)
    q_learner2 = PERLQLearning(**kargs)

    kargs['alpha'] = args.alpha2
    q_learner_target1 = PERLQLearning(**kargs)
    q_learner_target2 = PERLQLearning(**kargs)
    eps = 0

    s_raw_list = []
    s_raw = env.reset()
    s= obs_to_int_pi(s_raw, base=env.grid_size, raw_dim=raw_obs_dim)
    episode_rew = 0
    episode_rews =[]
    train_rews = []

    episode_count = 0
    episode_actions = []
    episode_states = []
    eval_rews = []
    n_actions_per_agent = env.action_space.nvec[0]
    n_new_state_action_list = []
    n_new_state_action = 0

    episode_step, total_step_count = 0, 0
    total_step_count_list = []
    episode_counts=[]

    while True:

        episode_step += 1
        total_step_count += 1

        alpha = episode_count / args.total_episode
        a1 = q_learner1.select_action(s, eps, q_learner_target1.q_table, alpha)
        a2 = q_learner2.select_action(s, eps, q_learner_target2.q_table, alpha)
        a = env.action_space.nvec[0] * a1 + a2
        episode_actions.append([a1, a2])
        episode_states.append(s)
        s_raw_list.append(s_raw)
        if meta_counter.count[s, a] == 0:
            n_new_state_action += 1
        meta_counter.update_count(s, a)
        s_next_raw, r, done = env.step([a1, a2])
        s_next= obs_to_int_pi(s_next_raw, base=env.grid_size, raw_dim=raw_obs_dim)
        q_learner1.insert_data(s, a1, r, s_next, done)
        q_learner2.insert_data(s, a2, r, s_next, done)
        q_learner_target1.insert_data(s, a1, r, s_next, done)
        q_learner_target2.insert_data(s, a2, r, s_next, done)
        meta_q.insert_data(s, a, r, s_next, done)
        episode_rew += r
        s = s_next
        s_raw = s_next_raw

        if done:
            episode_count += 1
            episode_counts.append(episode_count)
            episode_rews.append(episode_rew)
            if math.ceil(episode_count / N) <= raw_obs_dim:
                i = math.ceil(episode_count / N)
            else:
                i = raw_obs_dim
            sts=[]
            ats=[]
            s_a=[]
            data=meta_q.replay_buffer.get_random_batch(bz=1024)
            for s, a, r, ns, done in data:
                sts.append(s)
                ats.append(a)
                au1,au2=divmod(a, n_actions_per_agent)
                s_a.append(int_to_obs(s, base, raw_obs_dim)[0].tolist() + [au1,au2])
            s_m=np.mat(s_a).T
            p = meta_q.get_P(i, s_m)
            s_p = p*s_m
            s_p = s_p.astype(np.int32)
            L = s_p.T.tolist()
            gl = []
            first_min = min(L, key=L.count)
            for j in L:
                if L.count(j) == L.count(first_min) and j not in gl:
                    gl.append(j)
            goal = random.choice(gl)
            aug_index = L.index(goal)
            goal_s = sts[aug_index]
            goal_a1=s_a[aug_index][raw_obs_dim]
            goal_a2=s_a[aug_index][raw_obs_dim+1]


            q_learner_target1.update_q_from_D(reset_q=False)
            q_learner_target2.update_q_from_D(reset_q=False)
            q_learner1.update_q_from_D(goal=(goal_s,goal_a1))
            q_learner2.update_q_from_D(goal=(goal_s,goal_a2))
            meta_q.goal_q.clear()

            total_step_count_list.append(total_step_count)
            policy1, policy2 = q_learner_target1, q_learner_target2
            eval_rew = eval(policy1, policy2, eval_env, raw_obs_dim)
            eval_rews.append(eval_rew)
            train_rews.append(np.mean(episode_rews))

            data = {'episode': episode_counts, 'step': total_step_count_list,
                    'eval_rew': eval_rews, 'avg_rew': train_rews}
            dataframe = pd.DataFrame(data)
            dataframe.to_csv(os.path.join(log_path, 'push_box_dense_i_4.csv'),
                             columns=['episode', 'step', 'eval_rew', 'avg_rew'],
                             index=None)

            if episode_count % 20 == 0:
                print("episode {}, rew {}, avg. rew {}, n_new_state_action avg {:.4f}".
                      format(episode_count, episode_rew, np.mean(episode_rews),
                             np.mean(n_new_state_action_list[-100:])))

            n_new_state_action_list.append(n_new_state_action)
            n_new_state_action = 0
            episode_step = 0
            episode_actions = []
            episode_states = []
            s_raw_list = []

            episode_rew = 0
            s_raw = env.reset()
            s = obs_to_int_pi(s_raw, base=env.grid_size, raw_dim=raw_obs_dim)
            if episode_count >= args.total_episode:
                break

if __name__ == '__main__':
    main()
