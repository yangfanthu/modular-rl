import os
import gym
from gym.envs.registration import register
import wrappers
XML_DIR = './manipulation/xml'
ENV_DIR = './manipulation'
def registerEnvs(env_names, max_episode_steps, custom_xml):
    """register the MuJoCo envs with Gym and return the per-limb observation size and max action value (for modular policy training)"""
    # get all paths to xmls (handle the case where the given path is a directory containing multiple xml files)
    paths_to_register = []
    # existing envs
    if not custom_xml:
        for name in env_names:
            paths_to_register.append(os.path.join(XML_DIR, "{}.xml".format(name)))
    # custom envs
    else:
        if os.path.isfile(custom_xml):
            paths_to_register.append(custom_xml)
        elif os.path.isdir(custom_xml):
            for name in sorted(os.listdir(custom_xml)):
                if '.xml' in name:
                    paths_to_register.append(os.path.join(custom_xml, name))
    # register each env
    for xml in paths_to_register:
        env_name = os.path.basename(xml)[:-4]
        env_file = env_name
        # create a copy of modular environment for custom xml model
        params = {'xml': os.path.abspath(xml)}
        # register with gym
        register(id=("%s-v0" % env_name),
                 max_episode_steps=max_episode_steps,
                 entry_point="manipulation.%s:ModularEnv" % env_file,
                 kwargs=params)
        env = wrappers.IdentityWrapper(gym.make("%s-v0" % env_name))
        # the following is the same for each env
        limb_obs_size = env.limb_obs_size
        max_action = env.max_action
    return limb_obs_size, max_action

if __name__ == "__main__":
    # registerEnvs(['three_link_push'], 1000, False)
    # env = gym.make("three_link_push-v0")
    registerEnvs(['four_link_push'], 1000, False)
    env = gym.make("four_link_push-v0")
    # registerEnvs(['walker_3_main'], 1000, False)
    # env = gym.make("walker_3_main-v0")
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        reward,next_obs,done,info = env.step(action)
