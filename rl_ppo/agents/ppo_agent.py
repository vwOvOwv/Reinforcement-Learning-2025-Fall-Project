from .base_agent import MahjongGBAgent
import numpy as np
from collections import defaultdict, Counter

try:
    from MahjongGB import MahjongFanCalculator, MahjongShanten
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise

class PPOAgent(MahjongGBAgent):

    '''
    Observation Shape: 40 * 4 * 9
    ---------------------------------------------------------------------------
    [0]     SEAT_WIND           : 门风 (One-hot)
    [1]     PREVALENT_WIND      : 圈风 (One-hot)
    [2-5]   HAND                : 手牌 (4 channels: >=1, >=2, >=3, ==4)
    
    [6-8]   PACKS_SELF          : 自己的副露 (Chi, Pon, Gang)
    [9-11]  PACKS_DOWN          : 下家的副露 (Chi, Pon, Gang)
    [12-14] PACKS_OPP           : 对家的副露 (Chi, Pon, Gang)
    [15-17] PACKS_UP            : 上家的副露 (Chi, Pon, Gang)
    
    [18-21]    HISTORY_SELF     : 自己的弃牌 (4 channels: >=1, >=2, >=3, ==4)
    [22-25]    HISTORY_DOWN     : 下家的弃牌 (4 channels: >=1, >=2, >=3, ==4)
    [26-29]    HISTORY_OPP      : 对家的弃牌 (4 channels: >=1, >=2, >=3, ==4)
    [30-33]    HISTORY_UP       : 上家的弃牌 (4 channels: >=1, >=2, >=3, ==4)
    
    [34-37]    WALL_COUNT       : 牌墙剩余比例 (4 channels: >=1, >=2, >=3, ==4)
    [38]    SHANTEN             : 向听数
    [39]    LAST_PLAY           : 上一张被打出的牌
    [40]    VISIBLE             : 当前可看到的牌
    [44]    RECENT              : 最近两轮出的牌
    ---------------------------------------------------------------------------

    action_mask: 235
        pass1+hu1+discard34+chi63(3*7*3)+peng34+gang34+angang34+bugang34
    '''
    
    OBS_SIZE = 45
    ACT_SIZE = 235
    
    OFFSET_OBS = {
        'SEAT_WIND' : 0,
        'PREVALENT_WIND' : 1,
        'HAND' : 2,
        'PACKS': 6,
        'HISTORY': 18,
        'WALL': 34,
        'SHANTEN': 38,
        'LAST_PLAY': 39,
        'VISIBLE': 40,
        'RECENT': 44
    }
    OFFSET_ACT = {
        'Pass' : 0,
        'Hu' : 1,
        'Play' : 2,
        'Chi' : 36,
        'Peng' : 99,
        'Gang' : 133,
        'AnGang' : 167,
        'BuGang' : 201
    }
    TILE_LIST = [
        *('W%d'%(i+1) for i in range(9)),
        *('T%d'%(i+1) for i in range(9)),
        *('B%d'%(i+1) for i in range(9)),
        *('F%d'%(i+1) for i in range(4)),
        *('J%d'%(i+1) for i in range(3))
    ]
    OFFSET_TILE = {c : i for i, c in enumerate(TILE_LIST)}
    
    def __init__(self, seatWind):
        self.seatWind = seatWind
        self.packs = [[] for i in range(4)]
        self.history = [[] for i in range(4)]
        self.tileWall = [21] * 4
        self.shownTiles = defaultdict(int)
        self.wallLast = False
        self.isAboutKong = False
        self.last_played_tile = None
        self.recent_discards = []
        self.obs = np.zeros((self.OBS_SIZE, 36))
        self.obs[self.OFFSET_OBS['SEAT_WIND']][self.OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1
    
    ''' 环境发给agent的消息格式
    Wind 0..3
    Deal XX XX ...
    Player N Draw
    Player N Gang
    Player N(me) AnGang XX
    Player N(me) Play XX
    Player N(me) BuGang XX
    Player N(not me) Peng
    Player N(not me) Chi XX
    Player N(not me) AnGang
    
    Player N Hu
    Huang
    Player N Invalid
    Draw XX
    Player N(not me) Play XX
    Player N(not me) BuGang XX
    Player N(me) Peng
    Player N(me) Chi XX
    '''
    def request2obs(self, request):
        t = request.split()
        if t[0] == 'Wind':
            self.prevalentWind = int(t[1])
            self.obs[self.OFFSET_OBS['PREVALENT_WIND']][self.OFFSET_TILE['F%d' % (self.prevalentWind + 1)]] = 1
            return
        if t[0] == 'Deal':
            self.hand = t[1:]
            self._hand_embedding_update()
            return
        if t[0] == 'Huang':
            self.valid = []
            return self._obs()
        if t[0] == 'Draw':
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.valid = []
            self.last_played_tile = None
            if self._check_mahjong(tile, isSelfDrawn = True, isAboutKong = self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])
            self.isAboutKong = False
            self.hand.append(tile)
            self._hand_embedding_update()
            for tile in set(self.hand):
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                if self.hand.count(tile) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[tile])
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == 'PENG' and tile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[tile])
            return self._obs()
        # Player N Invalid/Hu/Draw/Play/Chi/Peng/Gang/AnGang/BuGang XX
        p = (int(t[1]) + 4 - self.seatWind) % 4
        if t[2] == 'Draw':
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return
        if t[2] == 'Invalid':
            self.valid = []
            return self._obs()
        if t[2] == 'Hu':
            self.valid = []
            return self._obs()
        if t[2] == 'Play':
            self.tileFrom = p
            self.curTile = t[3]
            self.last_played_tile = self.curTile
            self.recent_discards.append(self.curTile)
            if len(self.recent_discards) > 8:
                self.recent_discards.pop(0)
            self.shownTiles[self.curTile] += 1
            self.history[p].append(self.curTile)
            if p == 0:
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                return
            else:
                # Available: Hu/Gang/Peng/Chi/Pass
                self.valid = []
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[self.curTile])
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[self.curTile])
                    color = self.curTile[0]
                    if p == 3 and color in 'WTB':
                        num = int(self.curTile[1])
                        tmp = []
                        for i in range(-2, 3): tmp.append(color + str(num + i))
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 3) * 3 + 2)
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 2) * 3 + 1)
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 1) * 3)
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        if t[2] == 'Chi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            self.last_played_tile = None
            if p == 0:
                # Available: Play
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == 'UnChi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].pop()
            self.shownTiles[self.curTile] += 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            self.last_played_tile = None
            if p == 0:
                # Available: Play
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == 'UnPeng':
            self.packs[p].pop()
            self.shownTiles[self.curTile] -= 2
            if p == 0:
                for i in range(2):
                    self.hand.append(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3
            self.last_played_tile = None
            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True
            return
        if t[2] == 'AnGang':
            tile = 'CONCEALED' if p else t[3]
            self.packs[p].append(('GANG', tile, 0))
            self.last_played_tile = None
            if p == 0:
                self.isAboutKong = True
                for i in range(4):
                    self.hand.remove(tile)
                self.shownTiles[tile] = 4
            else:
                self.isAboutKong = False
            return
        if t[2] == 'BuGang':
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                    break
            self.shownTiles[tile] += 1
            self.last_played_tile = tile
            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
                return
            else:
                # Available: Hu/Pass
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn = False, isAboutKong = True):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        raise NotImplementedError('Unknown request %s!' % request)
    
    '''
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    '''
    def action2response(self, action):  # type: ignore
        if action < self.OFFSET_ACT['Hu']:
            return 'Pass'
        if action < self.OFFSET_ACT['Play']:
            return 'Hu'
        if action < self.OFFSET_ACT['Chi']:
            return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['Play']]
        if action < self.OFFSET_ACT['Peng']:
            t = (action - self.OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT['Gang']:
            return 'Peng'
        if action < self.OFFSET_ACT['AnGang']:
            return 'Gang'
        if action < self.OFFSET_ACT['BuGang']:
            return 'Gang ' + self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
        return 'BuGang ' + self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]
    
    '''
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    '''
    def response2action(self, response):
        t = response.split()
        if t[0] == 'Pass': return self.OFFSET_ACT['Pass']
        if t[0] == 'Hu': return self.OFFSET_ACT['Hu']
        if t[0] == 'Play': return self.OFFSET_ACT['Play'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Chi': return self.OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1
        if t[0] == 'Peng': return self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Gang': return self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'AnGang': return self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'BuGang': return self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT['Pass']
    
    def _obs(self):
        self._global_embedding_update()
        mask = np.zeros(self.ACT_SIZE)
        for a in self.valid:
            mask[a] = 1
        return {
            'observation': self.obs.reshape((self.OBS_SIZE, 4, 9)).copy(),
            'action_mask': mask
        }

    def _get_shanten(self):
        hand_list = list(self.hand)
        pack = tuple(self.packs[0]) 
        if len(hand_list) % 3 == 1:
            return MahjongShanten(pack=pack, hand=tuple(hand_list))
        else:
            min_s = 6
            unique_tiles = set(hand_list)
            for tile in unique_tiles:
                temp_hand = hand_list[:]
                temp_hand.remove(tile)
                
                s = MahjongShanten(pack=pack, hand=tuple(temp_hand))
                if s < min_s:
                    min_s = s
                if min_s <= 0: 
                    break
            return min_s
        
    def _hand_embedding_update(self):
        self.obs[self.OFFSET_OBS['HAND']: self.OFFSET_OBS['PACKS']] = 0 
        d = defaultdict(int)
        for tile in self.hand:
            d[tile] += 1
        for tile in d:
            self.obs[self.OFFSET_OBS['HAND'] : self.OFFSET_OBS['HAND'] + d[tile], self.OFFSET_TILE[tile]] = 1
    
    def _global_embedding_update(self):
        self.obs[self.OFFSET_OBS['PACKS']:] = 0
    
        # update packs (channel 6-17: Self, Down, Opp, Up x Chi, Peng, Gang)
        pack_type_map = {'CHI': 0, 'PENG': 1, 'GANG': 2}
        for i in range(4):
            base_idx = self.OFFSET_OBS['PACKS'] + (i * 3)
            for packType, tile, _ in self.packs[i]:
                if tile == 'CONCEALED':
                    continue
                ptype_offset = pack_type_map.get(packType, 0)
                channel_idx = base_idx + ptype_offset

                if packType in ['PENG', 'GANG']:
                    self.obs[channel_idx][self.OFFSET_TILE[tile]] = 1
                elif packType == 'CHI':
                    color = tile[0]
                    num = int(tile[1])
                    for offset in range(-1, 2):
                        t_name = color + str(num + offset)
                        if t_name in self.OFFSET_TILE:
                            self.obs[channel_idx][self.OFFSET_TILE[t_name]] = 1

        # update history (channel 18-33: Self, Down, Opp, Up) 
        for i in range(4):
            base_idx = self.OFFSET_OBS['HISTORY'] + (i * 4)
            hist_counter = Counter(self.history[i])
            for tile, count in hist_counter.items():
                if tile in self.OFFSET_TILE:
                    idx = self.OFFSET_TILE[tile]
                    if count >= 1: self.obs[base_idx + 0 + 0][idx] = 1
                    if count >= 2: self.obs[base_idx + 0 + 1][idx] = 1
                    if count >= 3: self.obs[base_idx + 0 + 2][idx] = 1
                    if count >= 4: self.obs[base_idx + 0 + 3][idx] = 1

        # update wall count (channel 34-37: Normalized)
        for i in range(4):
            self.obs[self.OFFSET_OBS['WALL'] + i] = self.tileWall[i] / 21.0

        # update shanten (channel 38)
        shanten = self._get_shanten()
        self.obs[self.OFFSET_OBS['SHANTEN']] = max(0, shanten + 1) / 6.0

        # update last play
        if self.last_played_tile and self.last_played_tile in self.OFFSET_TILE:
            self.obs[self.OFFSET_OBS['LAST_PLAY']][self.OFFSET_TILE[self.last_played_tile]] = 1

        # update visible tiles
        visible_counts = defaultdict(int)
        for i in range(4):
            for tile in self.history[i]:
                visible_counts[tile] += 1

        for i in range(4):
            for packType, tile, offer in self.packs[i]:
                if tile == 'CONCEALED': continue
                if packType == 'PENG': visible_counts[tile] += 3
                elif packType == 'GANG': visible_counts[tile] += 4
                elif packType == 'CHI':
                    color, num = tile[0], int(tile[1])
                    for offset in range(-1, 2):
                        t_name = color + str(num + offset)
                        visible_counts[t_name] += 1
        
        for tile in self.hand:
            visible_counts[tile] += 1

        base_vis = self.OFFSET_OBS['VISIBLE']
        for tile, count in visible_counts.items():
            if tile in self.OFFSET_TILE:
                idx = self.OFFSET_TILE[tile]
                if count >= 1: self.obs[base_vis + 0][idx] = 1
                if count >= 2: self.obs[base_vis + 1][idx] = 1
                if count >= 3: self.obs[base_vis + 2][idx] = 1
                if count >= 4: self.obs[base_vis + 3][idx] = 1

        # update recent discards
        base_rec = self.OFFSET_OBS['RECENT']
        self.obs[base_rec] = 0
        for tile in self.recent_discards:
            if tile in self.OFFSET_TILE:
                self.obs[base_rec][self.OFFSET_TILE[tile]] = 1
    
    def _check_mahjong(self, winTile, isSelfDrawn = False, isAboutKong = False):
        if isSelfDrawn:
            wall_last = (self.tileWall[0] == 0)
        else:
            wall_last = self.wallLast
        try:
            fans = MahjongFanCalculator(
                pack = tuple(self.packs[0]),
                hand = tuple(self.hand),
                winTile = winTile,
                flowerCount = 0,
                isSelfDrawn = isSelfDrawn,
                is4thTile = (self.shownTiles[winTile] + isSelfDrawn) == 4,
                isAboutKong = isAboutKong,
                isWallLast = wall_last,
                seatWind = self.seatWind,
                prevalentWind = self.prevalentWind,
                verbose = True
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8.0: raise Exception('Not Enough Fans')
        except:
            return False
        return True