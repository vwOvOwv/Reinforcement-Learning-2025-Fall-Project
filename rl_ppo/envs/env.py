from rl_ppo import MahjongGBAgent

import random
from collections import defaultdict

try:
    from MahjongGB import MahjongFanCalculator, MahjongShanten
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise

class Error(Exception):
    pass

class MahjongGBEnv():
    
    agent_names = ['player_%d' % i for i in range(1, 5)]
    
    def __init__(self, config):
        assert 'agent_clz' in config, "must specify agent_clz to process features!"
        self.agentclz = config['agent_clz']
        assert issubclass(self.agentclz, MahjongGBAgent), "ageng_clz must be a subclass of MahjongGBAgent!"

        self.duplicate = config.get('duplicate', True)
        self.variety = config.get('variety', -1)
        self.r = random.Random()
        self.reward_scaling = config.get('reward_scaling', None)
        self.observation_space = self.agentclz.observation_space
        self.action_space = self.agentclz.action_space

        self.all_tiles = []
        for suit in ['W', 'B', 'T']:
            for i in range(1, 10):
                self.all_tiles.append(suit + str(i))
        for suit in ['F']:
            for i in range(1, 5):
                self.all_tiles.append(suit + str(i))
        for suit in ['J']:
            for i in range(1, 4):
                self.all_tiles.append(suit + str(i))
    
    def reset(self, prevalentWind = -1, tileWall = ''):
        # Create agents to process features
        self.agents = [self.agentclz(i) for i in range(4)]
        self.reward = None
        self.done = False
        # Init random seed
        if self.variety > 0:
            random.seed(self.r.randint(0, self.variety - 1))
        # Init prevalent wind （圈风）
        self.prevalentWind = random.randint(0, 3) if prevalentWind < 0 else prevalentWind
        for agent in self.agents:
            agent.request2obs('Wind %d' % self.prevalentWind)
        # Prepare tile wall （牌墙）
        if tileWall:
            self.tileWall = tileWall.split()
        else:
            self.tileWall = []
            # 生成4副牌
            for j in range(4):
                # 1. 数牌 (1-9 万/饼/条)
                for i in range(1, 10):
                    self.tileWall.append('W' + str(i))  # Wan (万)
                    self.tileWall.append('B' + str(i))  # Bing (饼/筒)
                    self.tileWall.append('T' + str(i))  # Tiao (条/索)
                # 2. 风牌 (东/南/西/北)
                for i in range(1, 5):
                    self.tileWall.append('F' + str(i))
                # 3. 箭牌 (中/发/白)
                for i in range(1, 4):
                    self.tileWall.append('J' + str(i))
            random.shuffle(self.tileWall)
        self.originalTileWall = ' '.join(self.tileWall)
        if self.duplicate:
            self.tileWall = [self.tileWall[i * 34 : (i + 1) * 34] for i in range(4)]
        self.shownTiles = defaultdict(int)
        # Deal cards
        self._deal()

        # 计算初始向听数
        self.last_shanten = [self._get_shanten(i) for i in range(4)]
        self.last_tenpai = [False] * 4

        return self._obs()
    
    def step(self, action_dict):
        try:
            if self.state == 0:
                # After Chi/Peng, prepare to Play
                response = self.agents[self.curPlayer].action2response(action_dict[self.agent_names[self.curPlayer]]).split()
                if response[0] == 'Play':
                    self._discard(self.curPlayer, response[1])
                else:
                    raise Error(self.curPlayer)
                self.isAboutKong = False
            elif self.state == 1:
                # After Draw, prepare to Hu/Play/Gang/BuGang
                response = self.agents[self.curPlayer].action2response(action_dict[self.agent_names[self.curPlayer]]).split()
                if response[0] == 'Hu':
                    self._check_mahjong(self.curPlayer, isSelfDrawn = True, isAboutKong = self.isAboutKong)
                    self.shownTiles[self.curTile] += 1
                elif response[0] == 'Play':
                    self.hands[self.curPlayer].append(self.curTile)
                    self._discard(self.curPlayer, response[1])
                elif response[0] == 'Gang' and not self.myWallLast and not self.wallLast:
                    self._concealedKong(self.curPlayer, response[1])
                elif response[0] == 'BuGang' and not self.myWallLast and not self.wallLast:
                    self._promoteKong(self.curPlayer, response[1])
                else:
                    raise Error(self.curPlayer)
            elif self.state == 2:
                # After Play, prepare to Chi/Peng/Gang/Hu/Pass
                responses = {i : self.agents[i].action2response(action_dict[self.agent_names[i]]) for i in range(4) if i != self.curPlayer}
                t = {i : responses[i].split() for i in responses}
                # Priority: Hu > Peng/Gang > Chi
                for j in range(1, 4):
                    i = (self.curPlayer + j) % 4
                    if t[i][0] == 'Hu':
                        self._check_mahjong(i)
                        break
                else:
                    for j in range(1, 4):
                        i = (self.curPlayer + j) % 4
                        if t[i][0] == 'Gang' and self._canDrawTile(i) and not self.wallLast:
                            self._kong(i, self.curTile)
                            break
                        elif t[i][0] == 'Peng' and not self.wallLast:
                            self._pung(i, self.curTile)
                            break
                    else:
                        i = (self.curPlayer + 1) % 4
                        if t[i][0] == 'Chi' and not self.wallLast:
                            self._chow(i, t[i][1])
                        else:
                            for j in range(1, 4):
                                i = (self.curPlayer + j) % 4
                                if t[i][0] != 'Pass': raise Error(i)
                            if self.wallLast:
                                # A draw
                                self.obs = {i : self.agents[i].request2obs('Huang') for i in range(4)}
                                self.reward = [0, 0, 0, 0]
                                self.done = True
                            else:
                                # Next player
                                self.curPlayer = (self.curPlayer + 1) % 4
                                self._draw(self.curPlayer)
            elif self.state == 3:
                # After BuGang, prepare to Hu/Pass
                responses = {i : self.agents[i].action2response(action_dict[self.agent_names[i]]) for i in range(4) if i != self.curPlayer}
                for j in range(1, 4):
                    i = (self.curPlayer + j) % 4
                    if responses[i] == 'Hu':
                        self._check_mahjong(i, isAboutKong = True)
                        break
                else:
                    for j in range(1, 4):
                        i = (self.curPlayer + j) % 4
                        if responses[i] != 'Pass': raise Error(i)
                    self._draw(self.curPlayer)
        except Error as e:
            player = e.args[0]
            # err_msg = str(e)
            print(f"Player {player} Invalid Action.")
            print(f"State: {self.state}")
            print(f"Hand: {self.hands[player]}")
            print(f"Pack: {self.packs[player]}")
            print(f"Current tile: {self.curTile}")
            base_penalty = 10
            self.reward = [base_penalty] * 4
            self.reward[player] = -base_penalty * 3
            self.done = True
        return self._obs(), self._reward(), self._done()
        
    def _obs(self):
        return {self.agent_names[k] : v for k, v in self.obs.items()}

    def _get_shanten(self, player):
        hand_list = list(self.hands[player])
        pack = tuple(self.packs[player]) 

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

    def _reward(self):
        rewards = {name: 0.0 for name in self.agent_names}
        
        if self.reward:
            for k, val in enumerate(self.reward):
                rewards[self.agent_names[k]] += val

        for i in range(4):
            agent_name = self.agent_names[i]
            if self.reward_scaling:
                rewards[agent_name] /= self.reward_scaling

        return rewards
    
    def _done(self):
        return self.done
    
    def _drawTile(self, player):
        if self.duplicate:
            return self.tileWall[player].pop()
        return self.tileWall.pop()
    
    def _canDrawTile(self, player):
        if self.duplicate:
            return bool(self.tileWall[player])
        return bool(self.tileWall)
    
    def _deal(self):
        self.hands = []
        self.packs = []
        for i in range(4):
            hand = []
            while len(hand) < 13:
                tile = self._drawTile(i)
                hand.append(tile)
            self.hands.append(hand)
            self.packs.append([])
            self.agents[i].request2obs(' '.join(['Deal', *hand]))

        self.curPlayer = 0
        self.drawAboutKong = False
        self._draw(self.curPlayer)
    
    def _draw(self, player):
        tile = self._drawTile(player)
        self.myWallLast = not self._canDrawTile(player)
        self.wallLast = not self._canDrawTile((player + 1) % 4)
        self.isAboutKong = self.drawAboutKong
        self.drawAboutKong = False
        self.state = 1
        self.curTile = tile
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d Draw' % player)
        self.obs = {player : self.agents[player].request2obs('Draw %s' % tile)}
    
    def _discard(self, player, tile):   
        if tile not in self.hands[player]: raise Error(player)
        self.hands[player].remove(tile)
        self.shownTiles[tile] += 1
        self.wallLast = not self._canDrawTile((player + 1) % 4)
        self.curTile = tile
        self.state = 2
        self.agents[player].request2obs('Player %d Play %s' % (player, tile))
        self.obs = {i : self.agents[i].request2obs('Player %d Play %s' % (player, tile)) for i in range(4) if i != player}
    
    def _kong(self, player, tile):  
        self.hands[player].append(self.curTile) 
        if self.hands[player].count(tile) < 4: raise Error(player)
        for i in range(4): self.hands[player].remove(tile)
        # offer: 0 for self, 123 for up/oppo/down
        self.packs[player].append(('GANG', tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] = 4
        self.curPlayer = player
        self.drawAboutKong = True
        self.isAboutKong = False
        for agent in self.agents:
            agent.request2obs('Player %d Gang' % player)
        self._draw(player)
    
    def _pung(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 3: raise Error(player)
        for i in range(3): self.hands[player].remove(tile)
        # offer: 0 for self, 123 for up/oppo/down
        self.packs[player].append(('PENG', tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] += 2
        self.state = 0
        self.curPlayer = player
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d Peng' % player)
        self.obs = {player : self.agents[player].request2obs('Player %d Peng' % player)}
    
    def _chow(self, player, tile):
        self.hands[player].append(self.curTile)
        self.shownTiles[self.curTile] -= 1
        color = tile[0]
        num = int(tile[1])
        for i in range(-1, 2):
            t = color + str(num + i)
            if t not in self.hands[player]:
                raise Error(player)
            self.hands[player].remove(t)
            self.shownTiles[t] += 1
        # offer: 123 for which tile is offered
        self.packs[player].append(('CHI', tile, int(self.curTile[1]) - num + 2))
        self.state = 0
        self.curPlayer = player
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d Chi %s' % (player, tile))
        self.obs = {player : self.agents[player].request2obs('Player %d Chi %s' % (player, tile))}
    
    def _concealedKong(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 4: raise Error(player)
        for i in range(4): self.hands[player].remove(tile)
        # offer: 0 for self, 123 for up/oppo/down
        self.packs[player].append(('GANG', tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] = 4
        self.curPlayer = player
        self.drawAboutKong = True
        self.isAboutKong = False
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d AnGang' % player)
        self.agents[player].request2obs('Player %d AnGang %s' % (player, tile))
        self._draw(player)
    
    def _promoteKong(self, player, tile):
        self.hands[player].append(self.curTile)
        idx = -1
        for i in range(len(self.packs[player])):
            if self.packs[player][i][0] == 'PENG' and self.packs[player][i][1] == tile:
                idx = i
        if idx < 0: raise Error(player)
        self.hands[player].remove(tile)
        offer = self.packs[player][idx][2]
        self.packs[player][idx] = ('GANG', tile, offer)
        self.shownTiles[tile] = 4
        self.state = 3
        self.curPlayer = player
        self.curTile = tile
        self.drawAboutKong = True
        self.isAboutKong = False
        self.agents[player].request2obs('Player %d BuGang %s' % (player, tile))
        self.obs = {i : self.agents[i].request2obs('Player %d BuGang %s' % (player, tile)) for i in range(4) if i != player}
    
    def _check_mahjong(self, player, isSelfDrawn = False, isAboutKong = False):
        wall_last = self.myWallLast if isSelfDrawn else self.wallLast
        try:
            fans = MahjongFanCalculator(
                pack = tuple(self.packs[player]),
                hand = tuple(self.hands[player]),
                winTile = self.curTile,
                flowerCount = 0,
                isSelfDrawn = isSelfDrawn,
                is4thTile = (self.shownTiles[self.curTile] + isSelfDrawn) == 4,
                isAboutKong = isAboutKong,
                isWallLast = wall_last,
                seatWind = player,
                prevalentWind = self.prevalentWind,
                verbose = True
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8.0:
                raise Error('Not Enough Fans')
            self.obs = {i : self.agents[i].request2obs('Player %d Hu' % player) for i in range(4)}

            base_score = 8
            
            if isSelfDrawn:
                self.reward = [-(base_score + fanCnt)] * 4
                self.reward[player] = (base_score + fanCnt) * 3
            else:
                self.reward = [-base_score] * 4
                self.reward[player] = (base_score * 3) + fanCnt
                loser = self.curPlayer
                self.reward[loser] = -(base_score + fanCnt)

            self.done = True
        except Exception as e:
            print(f"SelfDrawn: {isSelfDrawn}, AboutKong: {isAboutKong}, WallLast: {wall_last}")
            print(f"Fan: {fanCnt}")
            raise Error(player)