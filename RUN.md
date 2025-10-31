


# python random_agent_v0.py --scenario basic --episodes 5 --rgb   --gif ./out/basic_v0/best.gif --gif_dir ./out/basic_v0/eps_gifs

python random_agent.py \
  --scenario basic \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 320x240 \
  --gif ./out/basic_v1/best.gif \
  --gif_dir ./out/basic_v1/eps \
  --fps 8 \
  --gif_scale 1 \
  --gif_repeat 2 \
  --seed 0