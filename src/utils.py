import json
import math
import os

import pandas as pd


def cosine_angle(x, y):
    num = 0
    a = 0
    b = 0
    for i in range(min([len(x), len(y)])):
        num += x[i] * y[i]
        a += x[i] ** 2
        b += y[i] ** 2
    num /= math.sqrt(a) * math.sqrt(b)
    return num


def euclidian_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


class Keys:
    outcome = 'outcome'
    playerId = 'playerId'
    position = 'position'
    teamId = 'teamId'
    timestamp = 'timestamp'
    type = 'type'
    date = 'date'
    home_team = 'home_team'
    away_team = 'away_team'
    events = 'events'


def get_player_names(path, filename):
    os.chdir(path)
    with open(filename) as f:
        tmp = json.load(f)
    names = {}
    for key, value in tmp.items():
        # print(value)
        names[int(key)] = value
    os.chdir('..')
    return names


def read_all(path, matches):
    os.chdir(path)
    for filename in os.listdir(os.getcwd()):
        with open(filename) as f:
            # print(filename)
            matches.append(json.load(f))
    os.chdir('..')


def write_type_outcomes(path, matches):
    type_outcome = []
    for m in matches:
        for e in m[Keys.events]:
            if Keys.outcome in e and e[Keys.outcome] not in type_outcome:
                type_outcome.append(e[Keys.outcome])
    os.chdir(path)
    with open('type_outcome', 'w') as f:
        f.writelines('\n'.join(type_outcome))
    os.chdir('..')


def write_type_events(path, matches):
    type_events = []
    for m in matches:
        for e in m[Keys.events]:
            if e[Keys.type] not in type_events:
                type_events.append(e[Keys.type])
    os.chdir(path)
    with open('type_events', 'w') as f:
        f.writelines('\n'.join(type_events))
    os.chdir('..')


def read_types(path, events, file):
    os.chdir(path)
    with open(file) as f:
        for line in f.readlines():
            events.append(line[:-1])
    os.chdir('..')


def write_matches(path, matches, file):
    os.chdir(path)
    i = 0
    for m in matches:
        m[Keys.events].sort(key=lambda x: x['timestamp'])
        filename = str(i) + file + '.json'
        with open(filename, 'w') as f:
            json.dump(m, f)
        i += 1
    os.chdir('..')


def write_events_csv(path, matches, file):
    os.chdir(path)
    i = 0
    for m in matches:
        m[Keys.events].sort(key=lambda x: x['timestamp'])
        for event in m[Keys.events]:
            # print(m[Keys.home_team], event[Keys.teamId])
            event[Keys.teamId] = 'home' if int(m[Keys.home_team]) == int(event[Keys.teamId]) \
                else 'away'
            if Keys.outcome in event:
                event[Keys.outcome] = str(event[Keys.outcome])
            event[Keys.playerId] = int(event[Keys.playerId])
            event[Keys.timestamp] = int(event[Keys.timestamp])
            event[Keys.type] = str(event[Keys.type])
        filename = str(i) + file + '.csv'
        pd.DataFrame(data=m[Keys.events]).to_csv(filename)
        i += 1
    os.chdir('../..')


def read_events_csv(path):
    frame = pd.DataFrame()
    os.chdir(path)
    for filename in os.listdir(os.getcwd()):
        new_frame = pd.read_csv(filename)
        frame = frame.append(new_frame)
    os.chdir('../..')
    return frame


def read_events_list(path):
    matches = []
    os.chdir(path)
    for filename in os.listdir(os.getcwd()):
        new_frame = pd.read_csv(filename)
        matches.append(new_frame)
        # break
    os.chdir('../..')
    return matches


if __name__ == '__main__':
    pass
