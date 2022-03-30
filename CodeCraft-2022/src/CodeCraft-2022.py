# -*- coding: utf-8 -*-
# @Time    : 2022/3/23 15:51
# @Author  : CYX
# @Email   : im.cyx@foxmail.com
# @File    : Code_test.py
# @Software: PyCharm
# @Project : CodeCraft-2022

import os
# import time
import numpy as np

# start = time.time()

# os.chdir("../../")
# directory = "./data"
# save_directory = "./output"

directory = "/data"
save_directory = "/output"

demand_src = "demand.csv"
apply_src = "site_bandwidth.csv"
qos_src = "qos.csv"
config_src = "config.ini"

save_file = "solution.txt"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)


def read_csv(path):
    self_ele, self_data = [], {}
    with open(path, "r", encoding='UTF-8') as f:
        reader = f.readlines()
        for i, row in enumerate(reader):
            row_ = row.strip("\n").split(",")
            if i == 0:
                self_ele = row_[1:]
            else:
                self_data.update({row_[0]: row_[1:]})
    return self_data, self_ele


def read_config(path):
    with open(path, "r") as f:
        res = f.readlines()[-1]
        return res.split('=')[-1]


demand, demand_user_id = read_csv(os.path.join(directory, demand_src))
apply, _ = read_csv(os.path.join(directory, apply_src))
qos, qos_user_id = read_csv(os.path.join(directory, qos_src))
qos_constraint = int(read_config(os.path.join(directory, config_src)))

demand_keys = np.array([item for item in demand.keys()], dtype=np.str)
demand = np.array([list(item) for item in demand.values()], dtype=np.int)
qos_keys = np.array([item for item in qos.keys()], dtype=np.str)
qos = np.array([list(item) for item in qos.values()], dtype=np.int)

# 输出到最终output
def output(final_res):
    with open(os.path.join(save_directory, save_file), "w+") as f:
        for single_time, single_time_res_dict in zip(final_res.keys(), final_res.values()):
            for single_user_name, single_user_value in zip(single_time_res_dict.keys(), single_time_res_dict.values()):
                f.write(f"{demand_user_id[single_user_name]}:")
                flag = 0
                for i, every_node_res in enumerate(single_user_value):
                    every_node_index = every_node_res[0]
                    every_node_value = every_node_res[1]
                    # print(every_node_index, every_node_value)
                    # 如果分配量为0,无需写入
                    if every_node_value == 0:
                        continue
                    else:
                        # 第一个分配前无需","
                        if not flag:
                            flag = 1
                        else:
                            f.write(",")
                        f.write(f"<{qos_keys[every_node_index]},{every_node_value}>")
                f.write("\n")


# 计算有效用户节点
def find_useful_server():
    # 挑选出满足每个客户带宽需求的边缘节点
    def limit_server_qos(axis_input):
        index_temp = np.where(qos[:, axis_input] < qos_constraint)
        qos_temp = np.array((index_temp[0], qos[index_temp][:, axis_input])).T
        qos_temp = qos_temp[np.argsort(qos_temp[:, 1], axis=0)]
        return qos_temp

    # 每个用户满足的节点列表
    # 每个用户满足的节点数量排序
    _client_limit_res, _client_limit_num_sequence, _client_limit_band_sequence = [], [], []
    for i in range(len(demand[0])):
        _server_res = limit_server_qos(i)
        _client_limit_res.append(_server_res)
        _client_limit_num_sequence.append(len(_server_res))
        _client_limit_band_sequence.append(sum([int(apply[qos_keys[x]][0]) for x in _server_res[:, 0]]))

    # 所有可用边缘节点集合
    _server_all = []
    # 构建所有可用边缘节点列表
    for x in _client_limit_res:
        _server_all += list(x[:, 0])
    _server_set = set(_server_all)

    # 每个客户所有满足的边缘节点的列表
    # 增加排序，（由于边缘节点时延目前无需考虑，直接按照序号排列），保证满足边缘节点搜索优先度
    _server_list = [sorted(list(_client_limit_res[i][:, 0])) for i in range(len(_client_limit_num_sequence))]

    # 每个客户拥有的满足边缘节点升序排序,首要排序指标数量,次要排序指标可提供带宽
    _server_num_sequence = np.argsort(np.array((_client_limit_num_sequence, _client_limit_band_sequence)).T, axis=0)
    # _server_num_sequence = [j[0] if j[0]==j[1] else j[1] for j in _server_num_sequence]
    _server_num_sequence = [j[0] for j in _server_num_sequence]
    # 排除完全没有边缘可以满足的用户
    while not len(_server_list[_server_num_sequence[0]]):
        _server_num_sequence = np.delete(_server_num_sequence, 0)

    # 对客户表进行排序
    temp_dict = {j: i for i, j in enumerate(_server_num_sequence)}
    _client_list = {j: [] for j in _server_set}
    for client, s_list in enumerate(_server_list):
        for s in s_list:
            _client_list[s].append(client)
    for k, cc in zip(_client_list.keys(), _client_list.values()):
        x = np.array([[c, temp_dict[c]] for c in cc])
        x = x[np.argsort(x[:, 1])][:, 0]
        _client_list[k] = x.tolist()

    # 对边缘节点按照可以被使用用户数量调整排序
    temp = []
    _client_num_sequence = []
    for y in _client_list.values():
        temp.append(len(y))
    temp_list = list(_server_set)
    for i, j in enumerate(np.argsort(temp)):
        _client_num_sequence.append(temp_list[j])
    _server_set = _client_num_sequence

    return _server_list, _server_num_sequence, _server_set, _client_list


# 边缘节点列表, 边缘节点数量队列, 边缘节点集合
server_list, server_num_sequence, server_set, client_wait_pool_max = find_useful_server()
# 按照流量统计各节点期望最大值
demand_avg = np.sum(demand,axis=0)/len(demand)
demand_avg = [int(avg // len(server_list[i])) for i, avg in enumerate(list(demand_avg))]
server_avg_list = {i: 0 for i in server_set}
res_free = {}

# # 循环检索边缘拥有用户列表
# for server, client_list in zip(client_wait_pool_max.keys(), client_wait_pool_max.values()):
#     # 计算当前用户手头用户列表各用户均分值之和 与 边缘节点上限 差值
#     free = sum(demand_avg[client] for client in client_list) - int(apply[qos_keys[server]][0])
#     # 如果超过节点上限
#     if free > 0:
#         # 期待均值最大值为上限
#         server_avg_list[server] = int(apply[qos_keys[server]][0])
#         # # 将多出来的值分摊各用户,并记住
#         # for client in client_list:
#         #     if client in res_free.keys():
#         #         res_free[client] += free // len(client_list)
#         #     else:
#         #         res_free.update({client: free // len(client_list)})
#         #     # demand_avg[client] += free // len(client_list)
#     # 否则加到各节点
#     else:
#         # 分配各边缘加到均值和
#         for client in client_list:
#             server_avg_list[server] += demand_avg[client]
#         # for client in client_list:
#         #     if client in res_free.keys():
#         #         if res_free[client] < free:
#         #             server_avg_list[server] += demand_avg[client]


# 前百分之5请求分界线(95%总请求次数向上取整)
demand_limit = len(demand) - int(np.ceil(len(demand) * 0.95)) - 50
# 各边缘使用量统计（超过临界点后不计）
server_total_use_bandwidth = {ii: [] for ii in server_set}
# 节点使用次数统计
server_total_use_num = {s: 0 for s in server_set}
# 等待分配边缘节点池(按平均分配策略)
server_max_avg = {i: 0 for i in server_set}
# 等待分配边缘节点池(按最大分配策略)
server_wait_pool_max = server_set.copy()

demand_test = demand.copy()

max_used_time = np.zeros(len(demand))

# 服务节点记录
time_used_server = [[] for i in range(len(demand))]

# 单次分配结果字典
total_res = {i: {j: [] for j in range(len(server_list))} for i in range(len(demand))}
# # 全局边缘节点带宽大小
# server_bandwidth = {node: int(apply[qos_keys[node]][0]) for node in server_set}

record_time = {s: [] for s in server_set}
for roll_time in range(demand_limit):
    for server_index in server_set:
        client_index_list = client_wait_pool_max[server_index]

        # user_demand_list 表示用户的需求数组, 第1列表示排序后的需求次序映射到原始次序的顺序, 第二列往后表示该边缘拥有的各用户按顺序的需求量
        user_demand_list = np.array([demand[:, index] for index in client_index_list]).T
        user_demand_list_sum = np.sum(user_demand_list, axis=1)
        user_sort_sequence = np.argsort(np.sum(user_demand_list, axis=0))

        user_demand_list = np.c_[np.arange(len(user_demand_list)), user_demand_list_sum, user_demand_list]

        user_demand_list = user_demand_list[np.argsort(user_demand_list[:,1], axis=0)[::-1]]


        for i, u_demand in enumerate(user_demand_list):
            if u_demand[0] not in record_time[server_index]:
                user_demand_list = user_demand_list[i]
                break

        user_demand = np.delete(user_demand_list, 1)
        time = user_demand[0]

        record_time[server_index].append(user_demand[0])


        # user_list_temp = {}
        # server_bandwidth = int(apply[qos_keys[server_index]][0])
        # time = user_demand[0]

        # # 用户数量
        # user_num = len(user_demand[1:])
        # # 最终分配结果
        # final_deliver = [0 for _ in range(user_num)]
        # # print(final_deliver)
        # src_demand = user_demand[1:]
        # # 如果边缘还有需求
        # while server_bandwidth:
        #     # 需要有限保证有解的用户池
        #     preserve_pool = []
        #     # 对用户建立分配列表
        #     deliver_list = [0 for demand in user_demand[1:]]
        #
        #     # 如果带宽超过了挑选用户总请求量的和
        #     if server_bandwidth > sum(user_demand[1:]):
        #         server_bandwidth -= sum(user_demand[1:])
        #         # 循环计算各个用户期望分配的均值
        #         for i in range(user_num):
        #             # 最终分配结果
        #             final_deliver[i] = user_demand[1+i]
        #             user_demand[i + 1] = 0
        #         break
        #
        #     # 循环计算各个用户期望分配的均值
        #     for i in user_sort_sequence:
        #         user = client_index_list[i]
        #         # 现在的需求
        #         # now_demand = user_demand[i+2]
        #         now_demand = src_demand[i]
        #         # 现在可用边缘节点集合
        #         now_set = set(server_list[user]) - set(time_used_server[time])
        #         # 当前理想分配值
        #         deliver_num = now_demand // len(now_set) + now_demand % len(now_set)
        #
        #         now_set = list(now_set)
        #         now_set.remove(server_index)
        #         # 剩余边缘可分配上限
        #         free_sum = sum(int(apply[qos_keys[i]][0]) for i in now_set)
        #
        #         # 最理想剩余依然超过当前分配数量
        #         if user_demand[i+1] / free_sum > 0.5:
        #             deliver_num = user_demand[i+1]
        #             preserve_pool.append(i)
        #         deliver_list[i] = deliver_num
        #
        #     # 如果边缘带宽上限大于挑选出的时刻用户以均值期望的请求总和
        #     if server_bandwidth > sum(deliver_list):
        #         # 剩下的边缘带宽
        #         server_bandwidth -= sum(deliver_list)
        #         # 更新最新的分配值
        #         for i in range(user_num):
        #             # 用户需求量更新
        #             user_demand[i+1] -= deliver_list[i]
        #             # 最终分配结果
        #             final_deliver[i] += deliver_list[i]
        #     # 如果不超过均值期望请求总和, 说明目前分配可以
        #     else:
        #         band = server_bandwidth
        #         for j in preserve_pool:
        #             # 如果分配的和已经超过上限
        #             if band == 0:
        #                 deliver_list[j] = 0
        #             elif band < deliver_list[j]:
        #                 deliver_list[j] = band
        #                 band = 0
        #             # 如果分配的和没超过上限
        #             else:
        #                 band -= deliver_list[j]
        #             print(deliver_list, band)
        #
        #         summary = sum(deliver_list)
        #
        #         # 模除标志
        #         free = server_bandwidth
        #         # 更新最新的分配值
        #         for i in user_sort_sequence:
        #             # 优先处理保护池子的用户
        #             if i in preserve_pool:
        #                 need = deliver_list[i]
        #                 free -= 0
        #             else:
        #                 # 分配比例
        #                 need = int(band * (deliver_list[i] / summary))
        #                 free -= need
        #             # 用户需求量更新
        #             user_demand[i+1] -= need
        #             # 最终分配结果
        #             final_deliver[i] += need
        #
        #         # 如果模没被分配
        #         if free < user_demand[0]:
        #             # 用户需求量更新
        #             user_demand[0] -= free
        #             # 最终分配结果
        #             final_deliver[0] += free
        #         # 更新的带宽值为0
        #         server_bandwidth = 0
        #
        #
        # # 最后写入结果
        # for index, user in enumerate(client_index_list):
        #     query = final_deliver[index]
        #     # 将边缘提供值写入当前时间各用户请求
        #     total_res[time][user].append([server_index, query])
        #     # 用户请求量更新
        #     demand[time, user] -= query
        #
        # # 记录当前时刻使用边缘节点
        # time_used_server[time].append(server_index)



        # 单个时间用户的需求
        user_list_temp = {}
        server_bandwidth = int(apply[qos_keys[server_index]][0])
        # 遍历各个用户
        for index, user in enumerate(client_index_list):
            # query: 该用户的请求
            query = user_demand[index + 1]
            # 如果这个用户的需求超过了当前服务器闲置带宽
            if query >= server_bandwidth:
                # 分配所有闲置带宽
                total_res[user_demand[0]][user].append([server_index, server_bandwidth])
                # 如果user_list_temp有东西，说明不是一个节点填满的，写进去
                for u, q in zip(user_list_temp.keys(), user_list_temp.values()):
                    # 保存分配
                    total_res[user_demand[0]][u].append([server_index, q])
                    # 用户请求量更新
                    demand[user_demand[0], u] = 0
                # 用户请求量更新
                demand[user_demand[0], user] -= server_bandwidth
                # 边缘服务量更新
                server_bandwidth = 0
                break
            else:
                # 服务器待分配带宽更新
                server_bandwidth -= query
                # 用户记录队列保存
                user_list_temp.update({user: query})

        if server_bandwidth != 0:
            for u, q in zip(user_list_temp.keys(), user_list_temp.values()):
                # 保存分配
                total_res[user_demand[0]][u].append([server_index, q])
                # 用户请求量更新
                demand[user_demand[0], u] = 0

        # 使用过的时间记数
        max_used_time[time] += 1

# 从小到大
demand_sequence = np.argsort([sum(d) for d in demand])


# 开始进行匹配
for num, demand_index in enumerate(demand_sequence):
    # 从时间序列上按用户需求总量降序匹配
    client_query_bandwidth = demand[demand_index]
    # 等待分配用户节点池
    client_wait_pool = list(server_num_sequence)
    # 等待分配边缘节点池
    server_wait_pool = server_wait_pool_max.copy()
    # # 单次分配结果字典
    # single_res = {i: [] for i in range(len(client_wait_pool))}
    # 全局边缘节点带宽大小
    server_bandwidth = {node: int(apply[qos_keys[node]][0]) for node in server_set}

    # 从用户角度迭代，满足用户未满足的需求

    # 挑选出此时刻不允许使用的边缘节点(已经被最大分配)
    server_useless = []
    for user_index, res_list in zip(total_res[demand_index].keys(), total_res[demand_index].values()):
        for server_list2 in res_list:
            server_useless.append(server_list2[0])

    for client in client_wait_pool:
        # 除了不能用的,剩下的加入列表
        use_server_index_list = []
        for s in server_list[client]:
            if s not in server_useless:
                use_server_index_list.append(s)
        # print(use_server_index_list)
        # print(use_server_index_list)
        query = client_query_bandwidth[client]

        server_temp = use_server_index_list.copy()
        # 迭代满足需求
        while query != 0:
            # # 计算均值策略可用各边缘节点使用率
            # use_rate = {node: (int(apply[qos_keys[node]][0]) - server_bandwidth[node]) / int(apply[qos_keys[node]][0])
            #             for node in use_server_index_list}
            # # 按照最大使用率差值计算各节点富余分配量
            # deliver_dict = {node: int((max(use_rate.values()) - use_rate[node]) * int(apply[qos_keys[node]][0]))
            #                 for node in use_server_index_list}
            deliver_dict = {}
            for node in use_server_index_list:
                if server_max_avg[node]:
                    deliver_dict.update({node: server_max_avg[node]})
                else:
                    # 计算均值策略可用各边缘节点使用率
                    use_rate = {
                        node: (int(apply[qos_keys[node]][0]) - server_bandwidth[node]) / int(apply[qos_keys[node]][0])
                        for node in use_server_index_list}
                    # 按照最大使用率差值计算各节点富余分配量
                    deliver_dict = {node: int((max(use_rate.values()) - use_rate[node]) * int(apply[qos_keys[node]][0]))
                                    for node in use_server_index_list}

            # 如果请求超过了当前动态分配的结果
            if query >= sum(deliver_dict.values()):
                res_query = query - sum(deliver_dict.values())
                # 平均分担当前待分配带宽
                mean_deliver = res_query // len(use_server_index_list)
                left_deliver = res_query % len(use_server_index_list)
                for node in use_server_index_list:
                    deliver_dict[node] += mean_deliver
            # 如果请求不超过当前动态分配的结果
            else:
                temp = sum(deliver_dict.values())
                for node in use_server_index_list:
                    deliver_dict[node] = int(query * (deliver_dict[node] / temp))
                left_deliver = query - sum(deliver_dict.values())
            # print(deliver_dict)
            # print(sum(deliver_dict.values()), query-sum(deliver_dict.values())-left_deliver)

            tempp = []
            for i in use_server_index_list:
                tempp.append(server_bandwidth[i])
            tempp = np.c_[use_server_index_list, tempp]
            tempp = tempp[np.argsort(tempp[:,1])][:,0][::-1]

            for use_server_index in tempp:
                free = server_bandwidth[use_server_index]
                if left_deliver > 0:
                    need = deliver_dict[use_server_index] + left_deliver
                    left_deliver = 0
                else:
                    need = deliver_dict[use_server_index]
                # # 如果待分配带宽为0
                # if free == 0:
                #     continue
                # 如果分配请求超过该节点可提供带宽上限
                if need > free:
                    # 存储单个分配节点信息
                    total_res[demand_index][client].append([use_server_index, free])
                    # 请求总量减小
                    query -= free
                    # 均值策略该节点待分配量归0
                    server_bandwidth[use_server_index] = 0
                    use_server_index_list.remove(use_server_index)
                else:
                    # 分配量为均值
                    total_res[demand_index][client].append([use_server_index, need])
                    # 请求总量减小
                    query -= need
                    # 均值策略该节点待分配量减小
                    server_bandwidth[use_server_index] -= need
        # print(server_bandwidth)
        for index, value in zip(server_bandwidth.keys(), server_bandwidth.values()):
            if index in server_temp:
                server_max_avg[index] = max(server_max_avg[index], int(apply[qos_keys[index]][0]) - value)

        # print(sum(server_max_avg.values()))
        # # 该用户需求已分配完, 退出等待分配池
        # client_wait_pool.pop(0)


    # def Calc_points():
    #     temp_res = {ii: 0 for ii in server_set}
    #     for time, value in zip(total_res.keys(), total_res.values()):
    #         for user, v in zip(value.keys(), value.values()):
    #             for res in v:
    #                 temp_res[res[0]] += res[1]
    #
    #     for index, num in zip(temp_res.keys(), temp_res.values()):
    #         server_total_use_bandwidth[index].append(num)
    #     for index in server_total_use_bandwidth.keys():
    #         server_total_use_bandwidth[index] = sorted(server_total_use_bandwidth[index], reverse=True) if len(
    #             server_total_use_bandwidth[index]) else []


    # Calc_points()

    # # if num < 10:
    # #     print(server_total_use_bandwidth[97])
    # final_res.append(single_res)
# print(server_total_use_bandwidth)




# print(server_total_use_bandwidth)

# print(total_res)

output(total_res)

# end = time.time()
# print(end - start)
