# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This file contains the commands to run PC-DAPS-1K (bag of 4 particles) with the pixel diffusion model on the FFHQ dataset.
# Metrics are reported as the mean over the 4 particles of every bag.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ++++ Nonlinear Tasks ++++
# phase retrieval
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=phase_retrieval \
+sampler=edm_pcdaps \
task_group=pixel \
save_dir=results/pcdaps/ffhq \
num_runs=1 \
batch_size=20 \
save_traj=False \
name=phase_retrieval \
gpu=0 &

# nonlinear deblur
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=nonlinear_blur \
+sampler=edm_pcdaps \
task_group=pixel \
save_dir=results/pcdaps/ffhq \
num_runs=1 \
batch_size=20 \
save_traj=False \
name=nonlinear_blur \
gpu=1 &

# high dynamic range
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=hdr \
+sampler=edm_pcdaps \
task_group=pixel \
save_dir=results/pcdaps/ffhq \
num_runs=1 \
batch_size=20 \
save_traj=False \
name=hdr \
gpu=2 &
PID_GPU2=$!

# ++++ Linear Tasks ++++
# super-resolution 4x
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=down_sampling \
+sampler=edm_pcdaps \
task_group=pixel \
save_dir=results/pcdaps/ffhq \
num_runs=1 \
batch_size=20 \
save_traj=False \
name=down_sampling \
gpu=3 &
PID_GPU3=$!

# super-resolution 16x
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=down_sampling_16x \
+sampler=edm_pcdaps \
task_group=pixel \
save_dir=results/pcdaps/ffhq \
num_runs=1 \
batch_size=20 \
save_traj=False \
name=down_sampling_16x \
gpu=4 &

# high-pass filtering
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=high_pass \
+sampler=edm_pcdaps \
task_group=pixel \
save_dir=results/pcdaps/ffhq \
num_runs=1 \
batch_size=20 \
save_traj=False \
name=high_pass \
gpu=5 &

# Gaussian deblur
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=gaussian_blur \
+sampler=edm_pcdaps \
task_group=pixel \
save_dir=results/pcdaps/ffhq \
num_runs=1 \
batch_size=20 \
save_traj=False \
name=gaussian_blur \
gpu=6 &

# motion deblur
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=motion_blur \
+sampler=edm_pcdaps \
task_group=pixel \
save_dir=results/pcdaps/ffhq \
num_runs=1 \
batch_size=20 \
save_traj=False \
name=motion_blur \
gpu=7 &

# the last two tasks start once GPU 2 and GPU 3 are free
wait $PID_GPU2

# box inpainting
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=inpainting \
+sampler=edm_pcdaps \
task_group=pixel \
save_dir=results/pcdaps/ffhq \
num_runs=1 \
batch_size=20 \
save_traj=False \
name=inpainting \
gpu=2 &
wait $PID_GPU3

# random inpainting
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=inpainting_rand \
+sampler=edm_pcdaps \
task_group=pixel \
save_dir=results/pcdaps/ffhq \
num_runs=1 \
batch_size=20 \
save_traj=False \
name=inpainting_rand \
gpu=3 &

wait
