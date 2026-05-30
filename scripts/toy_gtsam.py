import gtsam
import numpy as np
from gtsam.symbol_shorthand import X

# 4 keyframe'lik küçük bir döngü: 0 → 1 → 2 → 3 → 0 (loop)
graph = gtsam.NonlinearFactorGraph()
initial = gtsam.Values()

# Prior on first keyframe
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))
graph.add(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(), prior_noise))

# Initial estimates — sequential'da hata olsun bilerek
# Gerçek: 0 → (1,0,0) → (2,0,0) → (3,0,0) → (0,0,0)
# Tahmin: 0 → (1,0,0) → (2,0.1,0) → (3,0.2,0) — drift
initial.insert(X(0), gtsam.Pose3())
initial.insert(X(1), gtsam.Pose3(gtsam.Rot3(), np.array([1.0, 0.0, 0.0])))
initial.insert(X(2), gtsam.Pose3(gtsam.Rot3(), np.array([2.0, 0.1, 0.0])))
initial.insert(X(3), gtsam.Pose3(gtsam.Rot3(), np.array([3.0, 0.2, 0.0])))

# Sequential edges — her edge "ileri 1m" diyor
odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01]*6))
unit_forward = gtsam.Pose3(gtsam.Rot3(), np.array([1.0, 0.0, 0.0]))

graph.add(gtsam.BetweenFactorPose3(X(0), X(1), unit_forward, odom_noise))
graph.add(gtsam.BetweenFactorPose3(X(1), X(2), unit_forward, odom_noise))
graph.add(gtsam.BetweenFactorPose3(X(2), X(3), unit_forward, odom_noise))

# Loop closure: 3 → 0, "3 metre geri" 
loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05]*6))
loop_relative = gtsam.Pose3(gtsam.Rot3(), np.array([-3.0, 0.0, 0.0]))
graph.add(gtsam.BetweenFactorPose3(X(3), X(0), loop_relative, loop_noise))

# Optimize
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
result = optimizer.optimize()

# Sonuçları göster
for i in range(4):
    pose = result.atPose3(X(i))
    print(f"KF {i}: translation = {pose.translation()}")