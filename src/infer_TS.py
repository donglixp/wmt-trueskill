#!/usr/bin/env python
# encoding: utf-8

__author__ = "Keisuke Sakaguchi"
__version__ = "0.1"

# Input: JUDGEMENTS.csv which must contain one language-pair judgements.
# Output: *_mu_sigma.json: Mu and Sigma for each system
#        *.count: number of judgements among systems (for generating a heatmap) if -n is set to 2 and -e.

import sys
import os
import argparse
import random
import json
import numpy as np
import math
import itertools
import csv
import scripts.random_sample
import scripts.next_comparison
from itertools import combinations
from collections import defaultdict
from csv import DictReader
from trueskill import *

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('prefix', help='output ID (e.g. fr-en0)')
arg_parser.add_argument('-n', action='store', dest='freeN', type=int,
                        help='Free-for-All N (2-5)', default=2)
arg_parser.add_argument('-d', action='store', dest='dp', type=int,
                        help='Number of judgments to use (0 == all)', required=True)
arg_parser.add_argument('-p', action='store', dest='dp_pct', type=float, default=1.0,
                        help='Percentage of judgments to use (0.9)')
arg_parser.add_argument('-s', dest='num_systems', type=int, default=5,
                        help='Number of systems in one ranking in CSV file (default=5)')
arg_parser.add_argument('-e', dest='heat', default=False, action="store_true",
                        help='Produce a file for generating a heatmap (default=False)')
arg_parser.add_argument('-m', dest='metric_name', default="coh",
                        help='Metric name: coh/gram/red')
arg_parser.add_argument('-min', action='store', dest='work_time_min', type=float, default=20.0,
                        help='Min WorkTimeInSeconds')
arg_parser.add_argument('-max', action='store', dest='work_time_max', type=float, default=120.0,
                        help='Max WorkTimeInSeconds')

args = arg_parser.parse_args()

#######################################
### Global Variables and Parameters ###
param_sigma = 0.5
param_tau = 0.
draw_rate = 0.25

# You can set arbitrary number(s) for record (dp is the number assigned by -d).
#num_record = [int(args.dp*0.9), args.dp]
num_record = [args.dp]
#e.g. num_record = [args.dp*0.125, args.dp*0.25, args.dp*0.5, args.dp*0.9, args.dp]
#e.g. num_record = [400, 800, 1600, 3200, 6400, 11520, 12800]

# When -n is set to 2, you can set beginning and ending between (0 and 1) for counting the number of comparisons among systems.
# This is used for generating a heatmap.
# e.g. "count_begin=0.4 and count_end=0.6" records the number of comparisons from 40% to 60% of total comparisons.
count_begin = 0.8
count_end = 1.0
if count_begin > count_end:
    raise
#######################################

comparison_d = defaultdict(list)
mu_systems = [[], []]
sigma_systems = [[], []]


def parse_csv():
    # Parsing csv file and return system names and rank(1-5) for each sentence
    all_systems = set()
    for i, row in enumerate(DictReader(sys.stdin, delimiter='\t', quoting=csv.QUOTE_NONE)):
        all_systems.add(row.get('Input.system1'))
        all_systems.add(row.get('Input.system2'))

        work_time = float(row.get('WorkTimeInSeconds'))
        if work_time < args.work_time_min or work_time > args.work_time_max:
            continue
        system_tuple = (row.get('Input.system1'), row.get('Input.system2'))
        which_better = row.get("Answer.%s_better" % (args.metric_name)).lower()
        if which_better == 'a':
            rank_tuple = (0, 1)
        elif which_better == 'b':
            rank_tuple = (1, 0)
        else:
            continue
        comparison_d["_".join(tuple(sorted(set(system_tuple))))].append(
            (system_tuple, rank_tuple))
    return all_systems


def get_mu_sigma(sys_rate):
    sys_mu_sigma = {}
    for k, v in sys_rate.items():
        sys_mu_sigma[k] = [v.mu, v.sigma * v.sigma]
    return sys_mu_sigma


def sort_by_mu(sys_rate):
    sortlist = []
    for k, v in sys_rate.items():
        mu = v.mu
        sortlist.append((mu, k))
    sortlist.sort(reverse=True)
    return sortlist


def get_counts(s_name, c_dict, n_play):
    c_list = np.zeros((len(s_name), len(s_name)))
    total = sum(c_dict.values())
    for i, s_a in enumerate(s_name):
        for j, s_b in enumerate(s_name):
            c_list[i][j] = (c_dict[s_a + '_' + s_b] /
                            float(sum(c_dict.values()))) * 2
    return c_list.tolist()


def win_probability(team1, team2):
    delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
    sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
    size = len(team1) + len(team2)
    denom = math.sqrt(size * (BETA * BETA) + sum_sigma)
    ts = global_env()
    return ts.cdf(delta_mu / denom)


def get_sys_pair_win_prb(sys_rate):
    d = {}
    sys_name_list = sorted(sys_rate.keys())
    for k1 in sys_name_list:
        for k2 in sys_name_list:
            v1 = sys_rate[k1]
            v2 = sys_rate[k2]
            d[k1 + '-' + k2] = win_probability([v1], [v2])
    return d


def print_win_prb(sys_rate, d_win):
    sys_name_list = sorted(sys_rate.keys())
    print('%s\t%s' % ('name', '\t'.join(sys_name_list)))
    for k1 in sys_name_list:
        r_list = [d_win[k1 + '-' + k2] for k2 in sys_name_list]
        print('%s\t%s' %
              (k1, '\t'.join([str(it * 100.0)[:5] for it in r_list])))


def estimate_by_number():
    # Format of rating by one judgement:
    #  [[r1], [r2], [r3], [r4], [r5]] = rate([[r1], [r2], [r3], [r4], [r5]], ranks=[1,2,3,3,5])

    for num_iter_org in num_record:
        # setting for same number comparison (in terms of # of systems)
        inilist = [0] * args.freeN
        data_points = 0
        if num_iter_org == 0:
            # by # of pairwise judgements
            num_rankings = 0
            for key in comparison_d.keys():
                num_rankings += len(comparison_d[key])
            data_points = num_rankings / \
                len(list(combinations(inilist, 2))) + 1
        else:
            data_points = num_iter_org  # by # of matches
        num_iter = int(args.dp_pct * data_points)
        print >> sys.stderr, "Sampling %d / %d pairwise judgments" % (
            num_iter, data_points)
        param_beta = param_sigma * (num_iter / 40.0)
        env = TrueSkill(mu=0.0, sigma=param_sigma, beta=param_beta,
                        tau=param_tau, draw_probability=draw_rate)
        env.make_as_global()
        system_rating = {}
        num_play = 0
        counter_dict = defaultdict(int)
        for s in all_systems:
            system_rating[s] = Rating()
        while num_play < num_iter:
            num_play += 1
            systems_compared = scripts.next_comparison.get(
                get_mu_sigma(system_rating), args.freeN)
            systems_compared = "_".join(tuple(sorted(systems_compared)))
            # (systems, rank)
            obs = random.choice(comparison_d[systems_compared])
            systems_name_compared = obs[0]
            partial_rank = obs[1]

            if args.freeN == 2:
                if (num_play >= (num_iter * count_begin)) and (num_play <= (num_iter * count_end)):
                    sys_a = obs[0][0]
                    sys_b = obs[0][1]
                    counter_dict[sys_a + '_' + sys_b] += 1
                    counter_dict[sys_b + '_' + sys_a] += 1

            ratings = []
            for s in systems_name_compared:
                ratings.append([system_rating[s]])
            updated_ratings = rate(ratings, ranks=partial_rank)
            for s, r in zip(systems_name_compared, updated_ratings):
                system_rating[s] = r[0]

            if num_play == num_iter:
                f = open(args.prefix + '_mu_sigma.json', 'w')
                t = get_mu_sigma(system_rating)
                t['data_points'] = [data_points, args.dp_pct]
                t['win_prb'] = get_sys_pair_win_prb(system_rating)
                json.dump(t, f)
                f.close()

                # print_win_prb(system_rating, t['win_prb'])

                if (args.freeN == 2) and (num_iter_org == num_record[-1]) and args.heat:
                    f = open(args.prefix + '-' + str(count_begin) +
                             '-' + str(count_end) + '_count.json', 'w')
                    sys_names = zip(*sort_by_mu(system_rating))[1]
                    counts = get_counts(sys_names, counter_dict, num_play)
                    outf = {}
                    outf['sysname'] = sys_names
                    outf['counts'] = counts
                    json.dump(outf, f)
                    f.close()


if __name__ == '__main__':
    all_systems = parse_csv()
    estimate_by_number()
