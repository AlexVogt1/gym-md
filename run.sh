echo "------------------------------------------------------------------------"
echo "Job started on" `date`
echo "------------------------------------------------------------------------"
echo Running on $HOSTNAME...
echo Running on $HOSTNAME... >&2
# levels=("holmgard_9" "policy_0")
reward_test_levels=("check_1" "holmgard_9" "hard")
basic_play_styles=("treasure" "killer" "runner" "potion")
play_styles=("runner_safe" "runner_risky" "killer_safe" "killer_risky" "treasure_safe" "treasure_risky" "clearer_safe" "clearer_risky")
# source ~/.bashrc
# cd ~/src/EDPCGARL/gym-pcgrl
for i in "${reward_test_levels[@]}"; do   # The quotes are necessary here
    for j in "${basic_play_styles[@]}"; do   # The quotes are necessary here
        python train_ppo.py --env="$i" --play_style="$j" --reward_scheme=fifty_twoFifty --exp_type=rewardShaping
done
done

#finish hard exp
# fin_hard=("treasure_risky" "clearer_safe" "clearer_risky")
# for i in "${fin_hard[@]}"; do   # The quotes are necessary here
#     python train_ppo.py --env=hard --play_style="$i" --reward_scheme=not_original --exp_type=baseline
# done

#holmgard_9 original baseline
# python train_ppo.py --env="holmgard_9" --exp_type=baseline

#run with original reward
# for i in "${levels[@]}"; do   # The quotes are necessary here
#     python train_ppo.py --env="$i" --exp_type=baseline
# done

#train all other play_styles over all listed levels
# for i in "${levels[@]}"; do   # The quotes are necessary here
#     for j in "${play_styles[@]}"; do   # The quotes are necessary here
#         python train_ppo.py --env="$i" --play_style="$j" --reward_scheme=not_original --exp_type=baseline
# done
# done

# conda activate test1
# which python
# echo "Hello, we are doing a job now, seg smb, wide"
# pwd
# python train.py
echo "------------------------------------------------------------------------"
echo "Job ended on" `date`
echo "------------------------------------------------------------------------"