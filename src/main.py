import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils
from definitions import *
from utils import Keys
from utils import cosine_angle
from utils import euclidian_distance

InputDir = './data'
OutputDir = './output'
MetaDir = './meta'
SortedDir = './sorted_data'

matches = utils.read_events_list(SortedDir + '/events')
names = utils.get_player_names(MetaDir, 'players.json')
type_events = []
type_outcome = []
actions = []


def init_actions():
    for match in matches:
        players = []
        start = match.iloc[0][Keys.timestamp]
        team = match.iloc[0][Keys.teamId]
        for index, row in match.iterrows():
            if team != row[Keys.teamId]:
                actions.append(Action(players, start, players[-1].timestamp, team))
                start = row[Keys.timestamp]
                team = row[Keys.teamId]
                players = []
            player = Player(row[Keys.playerId], Move.get_move(row[Keys.type]),
                            Outcome.get_outcome(row[Keys.outcome]), row[Keys.timestamp])
            tmp = re.sub('[ \[\]\']', '', row[Keys.position]).split(',')
            positions = [float(x) for x in tmp] if tmp != [''] else []
            if len(positions) > 0:
                player.origin = (positions[0], positions[1])
            if len(positions) > 2:
                player.end = (positions[2], positions[3])
            players.append(player)
            # print(player)
        actions.append(Action(players, start, match.iloc[-1][Keys.timestamp], team))


def set_duration(actions):
    for i, action in enumerate(actions):
        for j, player in enumerate(action.players):
            if i == 0 and j == 0:
                player.duration = player.timestamp
            elif player == action.players[0]:
                # player.duration = actions[i-1].players[-1].timestamp - player.timestamp
                continue
            else:
                player.duration = player.timestamp - action.players[j - 1].timestamp
            player.duration += 1


def update_types():
    matches = []
    utils.read_all(InputDir, matches)
    utils.write_type_events(MetaDir, matches)
    utils.write_type_outcomes(MetaDir, matches)
    utils.write_events_csv(SortedDir + '\events', matches, '_events')


def filter_actions():
    return filter(lambda x: x.get_duration() != 0, actions)


velocity = {}


def avg_time(actions):
    """Calculates the average possession ball of the player during the actions"""
    for action in actions:
        for player in action.players[1:]:
            assert (player.duration >= 0)
            if player.move not in [Move.Pass, Move.Crosses, Move.Shot, Move.Aerieal]:
                continue
            name = names[player.id]
            time = float(action.get_duration()) / player.duration
            if name not in velocity.keys():
                velocity[name] = Velocity(time=(time, 1))
            else:
                velocity[name].set_time(time)


def avg_length(actions):
    """Average length of the pass normalized respect to the one that approaches more to the other team goal"""
    for action in actions:
        for j, player in enumerate(action.players[:-1]):
            if player.move not in [Move.Pass, Move.Crosses]:
                continue
            name = names[player.id]
            if player.end == () or action.players[j + 1].origin == ():
                continue
            x1, y1 = player.end
            x2, y2 = action.players[j + 1].origin
            xt, yt = (100, 50)
            length = euclidian_distance(x1, y1, x2, y2)
            angle = cosine_angle((x1, y1, x2, y2), (x1, y1, xt, yt))
            if name not in velocity.keys():
                velocity[name] = Velocity(length=(length, angle, 1))
            else:
                velocity[name].set_length(length, angle)


def avg_distance(actions):
    """Average distance traveled by a player when in possession with respect to the one that approaches more to the
    other team goal"""
    for action in actions:
        for player in action.players:
            if player.end == () or player.origin == ():
                continue
            name = names[player.id]
            x1, y1 = player.origin
            x2, y2 = player.end
            xt, yt = (100, 50)
            angle = cosine_angle((x1, y1, x2, y2), (x1, y1, xt, yt))
            distance = euclidian_distance(x1, y1, x2, y2)
            if name not in velocity.keys():
                velocity[name] = Velocity(distance=(distance, angle, 1))
            else:
                velocity[name].set_distance(distance, angle)


def avg_success(actions):
    """Average success rate of a player"""
    for action in actions:
        for player in action.players:
            if player.outcome == Outcome.Missing:
                continue
            name = names[player.id]
            success = 1
            if player.outcome in [Outcome.Failed, Outcome.Fouled, Outcome.Foul, Outcome.AerialFoul,
                                  Outcome.Failedclearance]:
                success = -1
            if player.outcome in [Outcome.Off_target]:
                success = 0
            if name not in velocity.keys():
                velocity[name] = Velocity(success=(success, 1))
            else:
                velocity[name].set_success(success)


def avg_position(actions):
    """Average position of a player"""
    for action in actions:
        for player in action.players:
            origin = None
            end = None
            if player.end == () and player.origin == ():
                continue
            name = names[player.id]
            if player.origin != ():
                origin = int(Position.get_position(player.origin))
            if player.end != ():
                end = int(Position.get_position(player.end))
            avg = (origin + end) / 2 if origin and end else None
            if avg is None:
                avg = origin if origin is not None else end
            if name not in velocity.keys():
                velocity[name] = Velocity(position=(avg, 1))
            else:
                velocity[name].set_position(avg)


def print_velocity():
    for key, value in velocity.items():
        print(key + ':', value)


def normalize_velocity():
    svd_time = np.std([x.get_avg_time() for x in velocity.values()])
    svd_length = np.std([x.get_avg_length() for x in velocity.values()])
    svd_distance = np.std([x.get_avg_distance() for x in velocity.values()])
    svd_success = np.std([x.get_avg_success() for x in velocity.values()])
    svd_position = np.std([x.get_avg_position() for x in velocity.values()])

    for value in velocity.values():
        time = value.get_avg_time() / svd_time
        length = value.get_avg_length() / svd_length
        distance = value.get_avg_distance() / svd_distance
        success = value.get_avg_success() / svd_success
        position = value.get_avg_position() / svd_position
        value.vector = (time, length, distance, success, position)


def print_vector():
    frame = pd.DataFrame(columns=['player', 'time', 'length', 'distance', 'success', 'position'])
    i = 0
    for key, value in velocity.items():
        frame.loc[i] = [key, *value.vector]
        i += 1
        print(key + ':', value.vector)
    frame.to_csv(OutputDir + '/vectors.csv')


def filter_velocity():
    delete_list = []
    for key, value in velocity.items():
        max_time = value.get_avg_time()
        max_length = value.get_avg_length()
        max_distance = value.get_avg_distance()
        max_success = value.get_avg_success()
        max_position = value.get_avg_position()
        if max_time < 0:
            delete_list.append(key)
        elif max_length < 0:
            delete_list.append(key)
        elif max_distance < 0:
            delete_list.append(key)
        elif max_success < 0:
            delete_list.append(key)
        elif max_position < 0:
            delete_list.append(key)
    for x in delete_list:
        velocity.pop(x)


def print_aggregate():
    players = []
    max_value = -1.
    for key, value in velocity.items():
        agg = 0.
        for x in value.vector:
            agg += x
        max_value = max(agg / 5, max_value)
        players.append((key, ':', agg / 5))
    players.sort(key=lambda x: x[2])
    players = [(x[0], x[1], (x[2] / max_value) * 100) for x in players]
    print(*[' '.join([str(y) for y in x]) for x in players], sep='\n')

    frame = pd.DataFrame(columns=['player', 'velocity'])
    for i, player in enumerate(players):
        frame.loc[i] = [player[0], player[2]]
    frame.to_csv(OutputDir + '/velocity.csv')


def histograms():
    frame = pd.DataFrame([x.get_avg_time() for x in velocity.values()])
    frame.plot(kind='hist')
    plt.title('time')
    plt.show()
    plt.close()
    frame = pd.DataFrame([x.get_avg_length() for x in velocity.values()])
    frame.plot(kind='hist')
    plt.title('length')
    plt.show()
    plt.close()
    frame = pd.DataFrame([x.get_avg_distance() for x in velocity.values()])
    frame.plot(kind='hist')
    plt.title('distance')
    plt.show()
    plt.close()
    frame = pd.DataFrame([x.get_avg_success() for x in velocity.values()])
    frame.plot(kind='hist')
    plt.title('success')
    plt.show()
    plt.close()
    frame = pd.DataFrame([x.get_avg_position() for x in velocity.values()])
    frame.plot(kind='hist')
    plt.title('position')
    plt.show()
    plt.close()


def main():
    # update_types()
    init_actions()
    filtered_actions = [x for x in actions if x.get_duration() > 0]
    set_duration(filtered_actions)
    avg_time(filtered_actions)
    avg_length(filtered_actions)
    avg_distance(filtered_actions)
    avg_success(actions)
    avg_position(actions)
    filter_velocity()
    # histograms()
    normalize_velocity()
    print_aggregate()
    print_vector()
    # print_vector()
    # print(*actions, sep='\n')
    # print_velocity()


if __name__ == '__main__':
    main()
