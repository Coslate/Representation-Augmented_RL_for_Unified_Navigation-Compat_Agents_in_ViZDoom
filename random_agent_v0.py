# random_agent.py
import argparse, random, numpy as np, imageio.v2 as imageio
import os
from tqdm import trange
import vizdoom as vzd
from doom_env import DoomEnv, _to_rgb

# ---- unified 12-action space (MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, ATTACK) ----
BUTTONS = [vzd.Button.MOVE_FORWARD, vzd.Button.TURN_LEFT,
           vzd.Button.TURN_RIGHT, vzd.Button.ATTACK]
ACTIONS = [
    [0,0,0,0],  # 0: noop
    [1,0,0,0],  # 1: forward
    [0,1,0,0],  # 2: turn left
    [0,0,1,0],  # 3: turn right
    [0,0,0,1],  # 4: shoot
    [1,1,0,0],  # 5: forward+left
    [1,0,1,0],  # 6: forward+right
    [1,0,0,1],  # 7: forward+shoot
    [0,1,0,1],  # 8: left+shoot
    [0,0,1,1],  # 9: right+shoot
    [1,1,0,1],  # 10: forward+left+shoot
    [1,0,1,1],  # 11: forward+right+shoot
]

def make_game(scenario: str, frame_repeat=4, rgb=True, res="320x240", seed=0):
    game = vzd.DoomGame()
    # Load stock scenarios from the pip package
    if scenario == "basic":
        cfg = os.path.join(vzd.scenarios_path, "basic.cfg")
    elif scenario in ("my_way_home", "mywayhome", "mwh"):
        cfg = os.path.join(vzd.scenarios_path, "my_way_home.cfg")
    else:
        raise ValueError("scenario must be one of: basic, my_way_home")

    game.load_config(cfg)

    # Headless / reproducible
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_seed(seed)

    # Screen format & resolution
    game.set_screen_format(vzd.ScreenFormat.RGB24 if rgb else vzd.ScreenFormat.GRAY8)
    res_map = {
        "320x240": vzd.ScreenResolution.RES_320X240,
        "160x120": vzd.ScreenResolution.RES_160X120,
        "800x600": vzd.ScreenResolution.RES_800X600,
    }
    game.set_screen_resolution(res_map.get(res, vzd.ScreenResolution.RES_320X240))

    # Override buttons to our unified set
    game.clear_available_buttons()
    for b in BUTTONS:
        game.add_available_button(b)

    game.set_mode(vzd.Mode.PLAYER)
    game.init()
    return game, frame_repeat

def run_random(game, frame_repeat, episodes=10, gif_path=None, fps=8, gif_dir=None, args=None):
    best_return = -1e9
    best_frames = []
    total_return = 0.0

    for i in trange(episodes, desc="Random episodes"):
        game.new_episode()
        ep_frames = []
        ep_ret = 0.0

        # track elapsed tics & last-known killcount
        ep_tics = 0
        ep_kills = 0
        ep_dead  = False

        # before game starting
        s0 = game.get_state()
        if s0 is not None and s0.screen_buffer is not None:
            arr0 = np.asarray(s0.screen_buffer)
            ep_frames.append(_to_rgb(arr0))

        while not game.is_episode_finished():
            a = random.randrange(len(ACTIONS))
            game.set_action(ACTIONS[a])
            step_r = 0.0
            for _ in range(frame_repeat):
                game.advance_action()
                step_r += game.get_last_reward()
                s = game.get_state()
                if s is not None and s.screen_buffer is not None:
                    ep_frames.append(_to_rgb(s.screen_buffer))
            ep_ret += step_r
            ep_tics += frame_repeat

        ep_dead  = ep_dead or game.is_player_dead()
        ep_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        secs = ep_tics / 35.0  # 35 tics per second
        total_return += ep_ret

        # infer reason
        secs = ep_tics / 35.0
        if ep_dead:
            reason = "death"
        elif (ep_kills >= 1 and args.scenario == "basic"):
            reason = "goal: kill"
        elif (args.scenario == "my_way_home" and ep_ret > 0):
            reason = "goal: reached vest"
        else:
            reason = "timeout"        

        print(f"Episode {i+1}/{episodes}: return={ep_ret:.2f}, KILLCOUNT={ep_kills}, time={secs:.1f}s ({ep_tics} tics), reason={reason}")

        if gif_dir and ep_frames:
            os.makedirs(gif_dir, exist_ok=True)
            imageio.mimsave(os.path.join(gif_dir, f"ep_{i+1:03d}_return_{ep_ret:.2f}.gif"), ep_frames, fps=fps)

        if ep_ret > best_return and len(ep_frames) > 0:
            best_return, best_frames = ep_ret, ep_frames

    avg_return = total_return / max(1, episodes)
    print(f"\nEpisodes: {episodes} | Average return: {avg_return:.2f} | Best: {best_return:.2f}")

    if gif_path and best_frames:
        out_dir = os.path.dirname(gif_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        root, ext = os.path.splitext(os.path.basename(gif_path))
        out_file = os.path.join(out_dir, f"{root}_return_{best_return:.2f}{ext}")
        imageio.mimsave(out_file, best_frames, fps=fps)
        print(f"Saved best-episode GIF to: {out_file}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=str, default="basic",
                   choices=["basic", "my_way_home"], help="ViZDoom scenario")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--frame_repeat", type=int, default=4)
    p.add_argument("--rgb", action="store_true", help="Use RGB instead of GRAY")
    p.add_argument("--gif", type=str, default="./out/best.gif")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif_dir", type=str, default=None)
    args = p.parse_args()

    game, fr = make_game(args.scenario, frame_repeat=args.frame_repeat, rgb=True if args.rgb else False, seed=args.seed)
    try:
        run_random(game, fr, episodes=args.episodes, gif_path=args.gif, fps=8, gif_dir=args.gif_dir, args=args)
    finally:
        game.close()

if __name__ == "__main__":
    main()
