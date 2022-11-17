from sc2.bot_ai import BotAI  # parent class we inherit from
from sc2.data import Difficulty, Race  # difficulty for bots, race for the 1 of 3 races
from sc2.main import run_game  # function that facilitates actually running the agents in games
from sc2.player import Bot, Computer  # wrapper for whether or not the agent is one of your bots, or a "computer" player
from sc2 import maps  # maps method for loading maps to play in.
from sc2.ids.unit_typeid import UnitTypeId
import random
import cv2
import math
import numpy as np
import sys
import pickle
import time

SAVE_REPLAY = True

total_steps = 10000
steps_for_pun = np.linspace(0, 1, total_steps)
step_punishment = ((np.exp(steps_for_pun ** 3) / 10) - 0.1) * 10


class IncrediBot(BotAI):  # inherits from BotAI (part of BurnySC2)
    async def on_step(self, iteration: int):  # on_step is a method that is called every step of the game.
        no_action = True
        while no_action:
            try:
                with open('state_rwd_action.pkl', 'rb') as f:
                    state_rwd_action = pickle.load(f)

                    if state_rwd_action['action'] is None:
                        # print("No action yet")
                        no_action = True
                    else:
                        # print("Action found")
                        no_action = False
            except:
                pass

        await self.distribute_workers()  # put idle workers back to work

        action = state_rwd_action['action']
        '''
        0: expand (ie: move to next spot, or build to 16 (minerals)+3 assemblers+3)
        1: build stargate (or up to one) (evenly)
        2: build voidray (evenly)
        3: send scout (evenly/random/closest to enemy?)
        4: attack (known buildings, units, then enemy base, just go in logical order.)
        5: voidray flee (back to base)
        '''

        # 0: expand (ie: move to next spot, or build to 16 (minerals)+3 assemblers+3)
        if action == 0:
            try:
                found_something = False
                if self.supply_left < 4:
                    # build pylons. 
                    if self.already_pending(UnitTypeId.PYLON) == 0:
                        if self.can_afford(UnitTypeId.PYLON):
                            await self.build(UnitTypeId.PYLON, near=random.choice(self.townhalls))
                            found_something = True

                if not found_something:

                    for nexus in self.townhalls:
                        # get worker count for this nexus:
                        worker_count = len(self.workers.closer_than(10, nexus))
                        if worker_count < 22:  # 16+3+3
                            if nexus.is_idle and self.can_afford(UnitTypeId.PROBE):
                                nexus.train(UnitTypeId.PROBE)
                                found_something = True

                        # have we built enough assimilators?
                        # find vespene geysers
                        for geyser in self.vespene_geyser.closer_than(10, nexus):
                            # build assimilator if there isn't one already:
                            if not self.can_afford(UnitTypeId.ASSIMILATOR):
                                break
                            if not self.structures(UnitTypeId.ASSIMILATOR).closer_than(2.0, geyser).exists:
                                await self.build(UnitTypeId.ASSIMILATOR, geyser)
                                found_something = True

                if not found_something:
                    if self.already_pending(UnitTypeId.NEXUS) == 0 and self.can_afford(UnitTypeId.NEXUS):
                        await self.expand_now()

            except Exception as e:
                print(e)

        # 1: build stargate (or up to one) (evenly)
        elif action == 1:
            try:
                # iterate thru all nexus and see if these buildings are close
                for nexus in self.townhalls:
                    # is there is not a gateway close:
                    if not self.structures(UnitTypeId.GATEWAY).closer_than(10, nexus).exists:
                        # if we can afford it:
                        if self.can_afford(UnitTypeId.GATEWAY) and self.already_pending(UnitTypeId.GATEWAY) == 0:
                            # build gateway
                            await self.build(UnitTypeId.GATEWAY, near=nexus)

                    # if the is not a cybernetics core close:
                    if not self.structures(UnitTypeId.CYBERNETICSCORE).closer_than(10, nexus).exists:
                        # if we can afford it:
                        if self.can_afford(UnitTypeId.CYBERNETICSCORE) and self.already_pending(
                                UnitTypeId.CYBERNETICSCORE) == 0:
                            # build cybernetics core
                            await self.build(UnitTypeId.CYBERNETICSCORE, near=nexus)

                    # if there is not a stargate close:
                    if not self.structures(UnitTypeId.STARGATE).closer_than(10, nexus).exists:
                        # if we can afford it:
                        if self.can_afford(UnitTypeId.STARGATE) and self.already_pending(UnitTypeId.STARGATE) == 0:
                            # build stargate
                            await self.build(UnitTypeId.STARGATE, near=nexus)

            except Exception as e:
                print(e)

        # 2: build VoidRay (random stargate)
        elif action == 2:
            try:
                if self.can_afford(UnitTypeId.VOIDRAY):
                    for sg in self.structures(UnitTypeId.STARGATE).ready.idle:
                        if self.can_afford(UnitTypeId.VOIDRAY):
                            sg.train(UnitTypeId.VOIDRAY)

            except Exception as e:
                print(e)

        # 3: send scout
        elif action == 3:
            # are there any idle probes:
            try:
                self.last_sent
            except:
                self.last_sent = 0

            # if self.last_sent doesnt exist yet:
            if (iteration - self.last_sent) > 200:
                try:
                    if self.units(UnitTypeId.PROBE).idle.exists:
                        # pick one of these randomly:
                        probe = random.choice(self.units(UnitTypeId.PROBE).idle)
                    else:
                        probe = random.choice(self.units(UnitTypeId.PROBE))
                    # send probe towards enemy base:
                    probe.attack(self.enemy_start_locations[0])
                    self.last_sent = iteration

                except Exception as e:
                    pass

        # 4: attack (known buildings, units, then enemy base, just go in logical order.)
        elif action == 4:
            try:
                # take all void rays and attack!
                for voidray in self.units(UnitTypeId.VOIDRAY).idle:
                    # if we can attack:
                    if self.enemy_units.closer_than(10, voidray):
                        # attack!
                        voidray.attack(random.choice(self.enemy_units.closer_than(10, voidray)))
                    # if we can attack:
                    elif self.enemy_structures.closer_than(10, voidray):
                        # attack!
                        voidray.attack(random.choice(self.enemy_structures.closer_than(10, voidray)))
                    # any enemy units:
                    elif self.enemy_units:
                        # attack!
                        voidray.attack(random.choice(self.enemy_units))
                    # any enemy structures:
                    elif self.enemy_structures:
                        # attack!
                        voidray.attack(random.choice(self.enemy_structures))
                    # if we can attack:
                    elif self.enemy_start_locations:
                        # attack!
                        voidray.attack(self.enemy_start_locations[0])

            except Exception as e:
                print(e)

        # 5: voidray flee (back to base)
        elif action == 5:
            if self.units(UnitTypeId.VOIDRAY).amount > 0:
                for vr in self.units(UnitTypeId.VOIDRAY):
                    vr.attack(self.start_location)

        # Construct the blank map
        map = np.zeros((self.game_info.map_size[0], self.game_info.map_size[1], 3), dtype=np.uint8)

        # draw the minerals:
        for mineral in self.mineral_field:
            pos = mineral.position
            c = [175, 255, 255]
            fraction = mineral.mineral_contents / 1800
            if mineral.is_visible:
                # print(mineral.mineral_contents)
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction * i) for i in c]
            else:
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [20, 75, 50]

        # draw the enemy start location:
        for enemy_start_location in self.enemy_start_locations:
            pos = enemy_start_location
            c = [0, 0, 255]
            map[math.ceil(pos.y)][math.ceil(pos.x)] = c

        # draw the enemy units:
        for enemy_unit in self.enemy_units:
            pos = enemy_unit.position
            c = [100, 0, 255]
            # get unit health fraction:
            fraction = enemy_unit.health / enemy_unit.health_max if enemy_unit.health_max > 0 else 0.0001
            map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction * i) for i in c]

        # draw the enemy structures:
        for enemy_structure in self.enemy_structures:
            pos = enemy_structure.position
            c = [0, 100, 255]
            # get structure health fraction:
            fraction = enemy_structure.health / enemy_structure.health_max if enemy_structure.health_max > 0 else 0.0001
            map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction * i) for i in c]

        # draw our structures:
        for our_structure in self.structures:
            # if it's a nexus:
            if our_structure.type_id == UnitTypeId.NEXUS:
                pos = our_structure.position
                c = [255, 255, 175]
                # get structure health fraction:
                fraction = our_structure.health / our_structure.health_max if our_structure.health_max > 0 else 0.0001
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction * i) for i in c]

            else:
                pos = our_structure.position
                c = [0, 255, 175]
                # get structure health fraction:
                fraction = our_structure.health / our_structure.health_max if our_structure.health_max > 0 else 0.0001
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction * i) for i in c]

        # draw the vespene geysers:
        for vespene in self.vespene_geyser:
            # draw these after buildings, since assimilators go over them. 
            # tried to denote some way that assimilator was on top, couldnt 
            # come up with anything. Tried by positions, but the positions arent identical. ie:
            # vesp position: (50.5, 63.5) 
            # bldg positions: [(64.369873046875, 58.982421875), (52.85693359375, 51.593505859375),...]
            pos = vespene.position
            c = [255, 175, 255]
            fraction = vespene.vespene_contents / 2250

            if vespene.is_visible:
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction * i) for i in c]
            else:
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [50, 20, 75]

        # draw our units:
        for our_unit in self.units:
            # if it is a voidray:
            if our_unit.type_id == UnitTypeId.VOIDRAY:
                pos = our_unit.position
                c = [255, 75, 75]
                # get health:
                fraction = our_unit.health / our_unit.health_max if our_unit.health_max > 0 else 0.0001
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction * i) for i in c]

            else:
                pos = our_unit.position
                c = [175, 255, 0]
                # get health:
                fraction = our_unit.health / our_unit.health_max if our_unit.health_max > 0 else 0.0001
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction * i) for i in c]

        # show map with opencv, resized to be larger:
        # horizontal flip:

        cv2.imshow('map', cv2.flip(cv2.resize(map, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST), 0))
        cv2.waitKey(1)

        if SAVE_REPLAY:
            # save map image into "replays dir"
            cv2.imwrite(f"replays/{int(time.time())}-{iteration}.png", map)

        reward = 0

        try:
            attack_count = 0
            # iterate through our void rays:
            for voidray in self.units(UnitTypeId.VOIDRAY):
                # if VoidRay is attacking and is in range of enemy unit:
                if voidray.is_attacking and voidray.target_in_range:
                    if self.enemy_units.closer_than(8, voidray) or self.enemy_structures.closer_than(8, voidray):
                        # reward += 0.005 # original was 0.005, decent results, but let's 3x it. 
                        reward += 0.015
                        attack_count += 1

        except Exception as e:
            print("reward", e)
            reward = 0

        if iteration % 100 == 0:
            print(f"Iter: {iteration}. RWD: {reward}. VR: {self.units(UnitTypeId.VOIDRAY).amount}")

        # write the file: 
        data = {"state": map, "reward": reward, "action": None, "done": False}  # empty action waiting for the next one!

        with open('state_rwd_action.pkl', 'wb') as f:
            pickle.dump(data, f)


result = run_game(  # run_game is a function that runs the game.
    maps.get("2000AtmospheresAIE"),  # the map we are playing on
    [Bot(Race.Protoss, IncrediBot()),  # runs our coded bot, protoss race, and we pass our bot object
     Computer(Race.Zerg, Difficulty.Hard)],  # runs a pre-made computer agent, zerg race, with a hard difficulty.
    realtime=False,  # When set to True, the agent is limited in how long each step can take to process.
)

if str(result) == "Result.Victory":
    rwd = 500
else:
    rwd = -500

with open("results.txt", "a") as f:
    f.write(f"{result}\n")

map = np.zeros((224, 224, 3), dtype=np.uint8)
observation = map
data = {"state": map, "reward": rwd, "action": None, "done": True}  # empty action waiting for the next one!
with open('state_rwd_action.pkl', 'wb') as f:
    pickle.dump(data, f)

cv2.destroyAllWindows()
cv2.waitKey(1)
time.sleep(3)
sys.exit()
