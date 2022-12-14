DIGITS = '0123456789abcdefghijklmnopqrstuvwxyz'
import numpy as np
import copy


def convert_to_base(decimal_number, base):  # 根据余数定位取DIGITS中的字符
    remainder_stack = []

    while decimal_number > 0:
        remainder = decimal_number % base
        remainder_stack.append(remainder)
        decimal_number = decimal_number // base

    new_digits = []
    while remainder_stack:
        new_digits.append(DIGITS[remainder_stack.pop()])

    return ''.join(new_digits)


def int_to_obs(int_rep, base, raw_dim):   # 把一个数值转化为6维状态,int_to_obs(8942512,15,6)=[[11 11 9 9 7 7]]
    """
    :param int_rep: int or 1-d numpy array
    :param base:
    :param raw_dim:
    :return: np array (bz, raw_dim)
    """
    if not isinstance(int_rep, np.ndarray):     # 判断int_rep是否为多维数组
        int_rep = np.array([int_rep])
    bz = int_rep.size
    obs = np.zeros((bz, raw_dim), dtype=np.int64)  # 生成bz x raw_dim大小的全零多维数组
    q = int_rep
    for d in range(raw_dim):
        q, r = divmod(q, base)   # q除以base,q为商，r为余
        obs[:, - (d + 1)] = r.reshape(-1)   # reshape()重新改变维数，reshape(-1)把r摆成1行
    if raw_dim == 5:
        obs_first_col = copy.deepcopy(obs[:, 0])
        obs[:, [0,1,2,3]] = obs[:, [1,2,3,4]]        # obs各列左移，第一列补到最后一列
        obs[:, -1] = obs_first_col
    return obs


def obs_to_int_pi(obs, base, raw_dim):    # 把6维状态编码/转化为一个数值 obs_to_int_pi([11 11 9 9 7 7],15,6)=8942512
    if raw_dim == 5:
        return obs[4] * base**4 + obs[0] * base**3 + obs[1] * base**2 + obs[2] * base + obs[3]
    elif raw_dim == 6:
        return obs[0] * base**5 + obs[1] * base**4 + obs[2] * base**3 + obs[3] * base**2 + obs[4] * base + obs[5]
    elif raw_dim == 7:
        return obs[0] * base ** 6 + obs[1] * base ** 5 + obs[2] * base ** 4 + \
               obs[3] * base ** 3 + obs[4] * base ** 2 + obs[5] * base + obs[6]
    elif raw_dim == 4:
        return obs[0] * base**3 + obs[1] * base**2 + obs[2] * base + obs[3]
    elif raw_dim == 3:
        return obs[0] * base**2 + obs[1] * base + obs[2]
    elif raw_dim == 2:
        return obs[0] * base + obs[1]
    elif raw_dim == 1:
        return obs[0]


def s_to_sp(s, base, raw_dim):        # s_to_sp(10405462,15,6)
    s = convert_to_base(s, base)      # s:string  s='da8177'
    s_p_str = str(s).zfill(raw_dim)   # 返回指定长度的字符串，原字符串右对齐，前面填充0
    if raw_dim == 5:
        s_p = int(s_p_str[0], base) * base**4 + int(s_p_str[3], base) * base**3 + int(s_p_str[4], base) * base**2 + int(s_p_str[1], base) * base + int(s_p_str[2], base)
    elif raw_dim == 6:
        s_p = int(s_p_str[2], base) * base**5 + int(s_p_str[3], base) * base**4 + int(s_p_str[0], base) * base**3 + int(s_p_str[1], base) * base**2 + int(s_p_str[4], base) * base + int(s_p_str[5], base)
    elif raw_dim == 7:
        s_p = int(s_p_str[2], base) * base**6 + int(s_p_str[3], base) * base**5 + int(s_p_str[0], base) * base**4 \
              + int(s_p_str[1], base) * base**3 + int(s_p_str[4], base) * base ** 2 + int(s_p_str[5], base) * base + int(s_p_str[6], base)
    return s_p   # raw_dim=6,s_p=6171862


def obs_to_level_int(raw_s, base, raw_dim, level, all_subspace):
    pass




