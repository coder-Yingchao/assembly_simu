from tianshou.data import Batch, ReplayBuffer, VectorReplayBuffer

# Assuming `replay_buffer` is your existing ReplayBuffer filled with data
# and `vector_replay_buffer` is your newly initialized VectorReplayBuffer

# Example initialization (adjust sizes as necessary)
filepath = './data/recorded_data0310.hdf5'
filepath2 = './data/recorded_data0310_2.hdf5'

replay_buffer = ReplayBuffer.load_hdf5(filepath, device='cuda')
vector_replay_buffer = ReplayBuffer(10000, 1)  # num_envs should match your environment setup

# Transfer data
for index in range(len(replay_buffer)):
    # Extract data by index
    # Prepare data in a format compatible with Tianshou's Batch
    data = replay_buffer[index]
    step_data = Batch(
        obs=data.obs,
        act=data.act+1,
        obs_next=data.obs_next,
        rew=data.rew,
        terminated=data.terminated, truncated=data.truncated
    )
    vector_replay_buffer.add(step_data)


vector_replay_buffer.save_hdf5(filepath2)

