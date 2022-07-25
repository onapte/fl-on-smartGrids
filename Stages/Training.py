# Input: The number of time steps T during one episode,
# training episodes E.
# Initialize RM , Q with random parameters θ .
# Initialize Q′ with parameters θ′.
# Initialize , increase and max .
# Output: The defending strategy.
# 1 for episode = 1 to E do
# 2 for step t = 1 to T do
# 3 Collect measurement vector yt ;
# 4 Estimate the state of system ˆxt by Equation (3)
# and Equation (4);
# 5 Compute the observation ot by Equation (12);
# 6 Compute the discrete observation value by
# Equation (13);
# 7  =  + increase;
# 8 if  > max then
# 9  = max ;
# 10 end
# 11 Generating a random number ra from (0, 1).
# 12 if ra <  then
# 13 at = a random action in action space.
# 14 else
# 15 at = argmaxQ(st , at , θ)
# 16 end
# 17 if st = sa then
# 18 if at = as then
# 19 rt = 0
# 20 else
# 21 rt = c1 ∗ |t − λ|
# 22 end
# 23 else
# 24 if at = as then
# 25 c2 ∗ ‖yt −h(ˆxt )‖
# ‖w‖
# 26 else
# 27 rt = 0
# 28 e

