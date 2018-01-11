from enum import Enum


class Outcome(Enum):
    Save = 1
    Blocked = 2
    Goal = 3
    Off_target = 4
    Wood_work = 5
    Success = 6
    Headed = 7
    Completed = 8
    Failed = 9
    Missing = -1
    Fouled = 11
    Assist = 13
    Foul = 14
    Clearance = 15
    Punch = 16
    Catch = 17
    AerialFoul = 18
    Failedclearance = 19
    Failedcatch = 20

    @staticmethod
    def get_outcome(outcome):
        outcome = str(outcome)
        if outcome == 'save':
            return Outcome.Save
        if outcome == 'blocked':
            return Outcome.Blocked
        if outcome == 'goal':
            return Outcome.Goal
        if outcome == 'off_target':
            return Outcome.Off_target
        if outcome == 'wood_work':
            return Outcome.Wood_work
        if outcome == 'success':
            return Outcome.Success
        if outcome == 'headed':
            return Outcome.Headed
        if outcome == 'completed':
            return Outcome.Completed
        if outcome == 'failed':
            return Outcome.Failed
        if outcome == 'Failed':
            return Outcome.Failed
        if outcome == 'Success':
            return Outcome.Success
        if outcome == 'Fouled':
            return Outcome.Fouled
        if outcome == 'Completed':
            return Outcome.Completed
        if outcome == 'Assist':
            return Outcome.Assist
        if outcome == 'Foul':
            return Outcome.Foul
        if outcome == 'clearance':
            return Outcome.Clearance
        if outcome == 'punch':
            return Outcome.Punch
        if outcome == 'catch':
            return Outcome.Catch
        if outcome == 'aerialFoul':
            return Outcome.AerialFoul
        if outcome == 'failedclearance':
            return Outcome.Failedclearance
        if outcome == 'failedcatch':
            return Outcome.Failedcatch
        if outcome == 'nan':
            return Outcome.Missing
        raise ValueError('Outcome unknown', outcome)


class Move(Enum):
    Save = 0
    Shot = 1
    Aerieal = 2
    Pass_block = 3
    Throw_away = 4
    Pass = 5
    Tackles = 6
    Crosses = 7
    Corners = 8
    Takeons = 9
    Fouls = 10
    Cards = 11

    @staticmethod
    def get_move(move):
        if move == 'save':
            return Move.Save
        elif move == 'pass_block':
            return Move.Pass_block
        elif move == 'tackles':
            return Move.Tackles
        elif move == 'takeons':
            return Move.Takeons
        elif move == 'crosses':
            return Move.Crosses
        elif move == 'aerieal':
            return Move.Aerieal
        elif move == 'throw_away':
            return Move.Throw_away
        elif move == 'corners':
            return Move.Corners
        elif move == 'shot':
            return Move.Shot
        elif move == 'goal_keeping':
            return Move.Goal_keeping
        elif move == 'fouls':
            return Move.Fouls
        elif move == 'cards':
            return Move.Cards
        elif move == 'pass':
            return Move.Pass
        else:
            raise ValueError('unrecognized move')


class Position(Enum):
    Defense = 1
    LowMid = 2
    Extern = 3
    HiMid = 4
    Wing = 5
    Attack = 6
    Undefined = 0

    @staticmethod
    def get_position(position):
        x, y = position
        if x < 55:
            return Position.Defense
        if x < 80.5 and (y < 10.84 * 2 or y > 40.16 * 2):
            return Position.Extern
        if x < 75:
            return Position.LowMid
        if x < 80.5:
            return Position.HiMid
        if 5.84 * 2 < y < 46.16 * 2:
            return Position.Attack
        if 5.84 * 2 >= y <= 46.16 * 2:
            return Position.Wing
        return Position.Undefined

    def __int__(self):
        if self == Position.Defense:
            return 0
        if self == Position.LowMid:
            return 3
        if self == Position.Extern:
            return 0
        if self == Position.HiMid:
            return 5
        if self == Position.Wing:
            return 1
        if self == Position.Attack:
            return 10
        if self == Position.Undefined:
            return 0

    def __add__(self, other):
        return int(self) + int(other)


class Player:
    def __init__(self, id, move, outcome, timestamp, origin=(), end=(), duration=-1):
        self.id = id
        self.move = move
        self.outcome = outcome
        self.origin = origin
        self.end = end
        self.timestamp = timestamp
        self.duration = duration

    def __repr__(self):
        return ' '.join(
            [str(x) for x in [self.id, self.move, self.outcome, self.timestamp, self.origin, self.end, self.duration]])


class Action:
    def __init__(self, players=None, start=0, end=0, team=None):
        # for player in players:
        #     player.timestamp-=start
        self.players = players
        self.start = start
        self.end = end
        self.team = team

    def __repr__(self):
        return ' '.join(['start:' + str(self.start), 'stop:' + str(self.end), 'duration:' + str(self.get_duration()),
                         'players:' + str(self.players), ''])

    def __len__(self):
        return len(self.players)

    def get_duration(self):
        return self.end - self.start


class Velocity:
    cutoff = 50

    def __init__(self, time=(0., 0), length=(0.0, 0., 0), distance=(0., 0., 0), success=(0, 0), position=(0, 0)):
        self.time = time
        self.length = length
        self.distance = distance
        self.success = success
        self.position = position
        self.vector = ()

    def set_time(self, time):
        x, y = self.time
        self.time = (x + time, y + 1)

    def set_length(self, length, angle):
        x, y, z = self.length
        self.length = (x + length, y + angle, z + 1)

    def set_distance(self, distance, angle):
        x, y, z = self.distance
        self.distance = (x + distance, y + angle, z + 1)

    def set_success(self, event):
        x, y = self.success
        self.success = (x + event, y + 1)

    def set_position(self, position):
        x, y = self.position
        self.position = (x + position, y + 1)

    def get_avg_time(self):
        x, y = self.time
        if y > Velocity.cutoff:
            return x / y
        return -1

    def get_avg_length(self):
        x, y, z = self.length
        if z > Velocity.cutoff:
            return x / z * y / z
        return -1

    def get_avg_distance(self):
        x, y, z = self.distance
        if z > Velocity.cutoff:
            return x / z * y / z
        return -1

    def get_avg_success(self):
        x, y = self.success
        if y > Velocity.cutoff:
            return x / y
        return -1

    def get_avg_position(self):
        x, y = self.position
        if y > self.cutoff:
            return x / y
        return -1

    def __repr__(self):
        return ' '.join([str(x) for x in [self.get_avg_time(), self.get_avg_length(), self.get_avg_distance(),
                                          self.get_avg_success(), self.get_avg_position()]])
